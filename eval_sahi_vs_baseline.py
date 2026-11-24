#!/usr/bin/env python3
"""
Fair evaluation of YOLO26n baseline vs. YOLO26n + SAHI on VisDrone2019-DET.

Why this exists: the repo previously reported SAHI mAP as `baseline + 6.42`
(hardcoded). This script MEASURES it, scoring every configuration with the
SAME COCO-style evaluator (pycocotools) so the numbers are comparable.

Run in Colab (GPU), after the notebook has downloaded VisDrone and converted
labels to YOLO format (Phase 1):

    !pip -q install ultralytics sahi pycocotools
    !python eval_sahi_vs_baseline.py --weights best.pt \
        --data-root /content/datasets/VisDrone \
        --split VisDrone2019-DET-val --limit 20          # quick smoke test
    !python eval_sahi_vs_baseline.py --weights best.pt \
        --data-root /content/datasets/VisDrone \
        --split VisDrone2019-DET-val --baseline-imgsz 640 1024
    !python eval_sahi_vs_baseline.py --weights best.pt \
        --data-root /content/datasets/VisDrone \
        --split VisDrone2019-DET-test-dev

Notes / caveats (state these in the report):
  * Ground truth = converted YOLO labels (ignored regions and category 11
    removed), so predictions inside ignored regions count as false positives.
    Numbers can differ slightly from Ultralytics' own val() and from the
    official VisDrone toolkit. Compare configurations ONLY within this script's
    output, and label the protocol "COCO-style, pycocotools, maxDets=500".
  * Accuracy passes use conf=0.001 (needed for mAP). Precision/recall use a
    fixed operating point (conf=0.25, IoU=0.5, class-aware greedy matching).
  * Timing is a separate pass at conf=0.25, end-to-end per image (disk read
    included), after warm-up, on whatever GPU this runs on.
  * Baseline at imgsz>640 here is the 640-trained model run at higher
    resolution (cheap check). A fully fair high-res baseline needs fine-tuning
    at that resolution; say so if you report it.
"""
import argparse
import json
import platform
import time
from collections import defaultdict
from pathlib import Path

CLASS_NAMES = ['pedestrian', 'people', 'bicycle', 'car', 'van',
               'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
IMG_EXTS = {'.jpg', '.jpeg', '.png'}
MAX_DETS = 500


# ----------------------------------------------------------------------------
# Pure-python helpers (no heavy imports, easy to unit test)
# ----------------------------------------------------------------------------
def list_images(img_dir, limit=None):
    imgs = sorted(p for p in Path(img_dir).iterdir() if p.suffix.lower() in IMG_EXTS)
    return imgs[:limit] if limit else imgs


def read_size(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size  # (w, h)


def yolo_labels_to_boxes(label_path, w, h):
    """YOLO normalized (cls xc yc w h) -> list of (cls, x, y, bw, bh) in pixels."""
    boxes = []
    label_path = Path(label_path)
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        c = int(p[0])
        xc, yc, bw, bh = (float(v) for v in p[1:5])
        bw, bh = bw * w, bh * h
        boxes.append((c, xc * w - bw / 2, yc * h - bh / 2, bw, bh))
    return boxes


def build_coco_gt(images, label_dir):
    gt = {'images': [], 'annotations': [],
          'categories': [{'id': i + 1, 'name': n} for i, n in enumerate(CLASS_NAMES)]}
    ann_id = 1
    for i, p in enumerate(images):
        w, h = read_size(p)
        gt['images'].append({'id': i, 'file_name': p.name, 'width': w, 'height': h})
        for c, x, y, bw, bh in yolo_labels_to_boxes(Path(label_dir) / (p.stem + '.txt'), w, h):
            if bw <= 0 or bh <= 0:
                continue
            gt['annotations'].append({'id': ann_id, 'image_id': i, 'category_id': c + 1,
                                      'bbox': [x, y, bw, bh], 'area': bw * bh, 'iscrowd': 0})
            ann_id += 1
    return gt


def iou_xywh(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def precision_recall_at(gt, preds, conf=0.25, iou_thr=0.5):
    """Class-aware greedy matching per image at a fixed confidence threshold."""
    gts = defaultdict(list)
    for a in gt['annotations']:
        gts[(a['image_id'], a['category_id'])].append(a['bbox'])
    dts = defaultdict(list)
    for d in preds:
        if d['score'] >= conf:
            dts[(d['image_id'], d['category_id'])].append(d)
    tp = fp = 0
    for key, dl in dts.items():
        g = gts.get(key, [])
        used = [False] * len(g)
        for d in sorted(dl, key=lambda d: -d['score']):
            best, best_iou = -1, iou_thr
            for j, gb in enumerate(g):
                if used[j]:
                    continue
                v = iou_xywh(d['bbox'], gb)
                if v >= best_iou:
                    best, best_iou = j, v
            if best >= 0:
                used[best] = True
                tp += 1
            else:
                fp += 1
    n_gt = len(gt['annotations'])
    return {'precision': tp / (tp + fp) if tp + fp else 0.0,
            'recall': tp / n_gt if n_gt else 0.0, 'tp': tp, 'fp': fp, 'n_gt': n_gt}


# ----------------------------------------------------------------------------
# Prediction + evaluation
# ----------------------------------------------------------------------------
def predict_baseline(model, images, imgsz, conf, device):
    preds = []
    for i, p in enumerate(images):
        r = model.predict(source=str(p), imgsz=imgsz, conf=conf, iou=0.7,
                          max_det=MAX_DETS, device=device, verbose=False)[0]
        b = r.boxes
        if b is None or len(b) == 0:
            continue
        xyxy = b.xyxy.cpu().numpy()
        sc = b.conf.cpu().numpy()
        cl = b.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), s, c in zip(xyxy, sc, cl):
            preds.append({'image_id': i, 'category_id': int(c) + 1,
                          'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                          'score': float(s)})
    return preds


def predict_sahi(det_model, images, slice_size, overlap):
    from sahi.predict import get_sliced_prediction
    preds = []
    for i, p in enumerate(images):
        res = get_sliced_prediction(
            str(p), det_model,
            slice_height=slice_size, slice_width=slice_size,
            overlap_height_ratio=overlap, overlap_width_ratio=overlap,
            postprocess_type='GREEDYNMM', postprocess_match_metric='IOS',
            postprocess_match_threshold=0.5, verbose=0)
        for op in res.object_prediction_list:
            bb = op.bbox
            preds.append({'image_id': i, 'category_id': int(op.category.id) + 1,
                          'bbox': [float(bb.minx), float(bb.miny),
                                   float(bb.maxx - bb.minx), float(bb.maxy - bb.miny)],
                          'score': float(op.score.value)})
    return preds


def coco_eval(gt_dict, preds):
    import contextlib
    import io
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    empty = {'mAP50': 0.0, 'mAP50_95': 0.0,
             'per_class_AP50': {n: 0.0 for n in CLASS_NAMES},
             'per_class_AP50_95': {n: 0.0 for n in CLASS_NAMES}}
    if not preds:
        return empty
    quiet = io.StringIO()
    gt = COCO()
    gt.dataset = gt_dict
    with contextlib.redirect_stdout(quiet):
        gt.createIndex()
        dt = gt.loadRes(preds)
        E = COCOeval(gt, dt, 'bbox')
        E.params.maxDets = [1, 10, MAX_DETS]
        E.evaluate()
        E.accumulate()
    prec = E.eval['precision']  # [T, R, K, A, M]

    def mean_valid(a):
        a = a[a > -1]
        return float(a.mean()) if a.size else 0.0

    out = {'mAP50': mean_valid(prec[0, :, :, 0, -1]) * 100,
           'mAP50_95': mean_valid(prec[:, :, :, 0, -1]) * 100,
           'per_class_AP50': {}, 'per_class_AP50_95': {}}
    for k, name in enumerate(CLASS_NAMES):
        out['per_class_AP50'][name] = mean_valid(prec[0, :, k, 0, -1]) * 100
        out['per_class_AP50_95'][name] = mean_valid(prec[:, :, k, 0, -1]) * 100
    return out


# ----------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------
def time_ms(fn, images, warmup=3):
    try:
        import torch
        sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    except ImportError:
        sync = lambda: None  # noqa: E731
    for p in images[:warmup]:
        fn(p)
    sync()
    ts = []
    for p in images:
        t0 = time.perf_counter()
        fn(p)
        sync()
        ts.append((time.perf_counter() - t0) * 1000)
    return sum(ts) / len(ts)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='best.pt')
    ap.add_argument('--data-root', default='/content/datasets/VisDrone')
    ap.add_argument('--split', default='VisDrone2019-DET-val')
    ap.add_argument('--limit', type=int, default=None, help='use first N images (smoke test)')
    ap.add_argument('--baseline-imgsz', type=int, nargs='+', default=[640])
    ap.add_argument('--slice', type=int, default=640)
    ap.add_argument('--overlap', type=float, default=0.2)
    ap.add_argument('--time-n', type=int, default=100, help='images used for timing')
    ap.add_argument('--device', default='0')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    import importlib.metadata as md
    import torch
    from sahi import AutoDetectionModel
    from ultralytics import YOLO

    split_dir = Path(args.data_root) / args.split
    images = list_images(split_dir / 'images', args.limit)
    if not images:
        raise SystemExit(f'No images found in {split_dir / "images"}')
    print(f'Split {args.split}: {len(images)} images')
    gt = build_coco_gt(images, split_dir / 'labels')
    print(f'Ground truth: {len(gt["annotations"])} boxes')

    use_cuda = torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(0) if use_cuda else 'CPU'
    sahi_device = 'cuda:0' if use_cuda else 'cpu'
    model = YOLO(args.weights)

    def make_sahi(conf):
        for mt in ('ultralytics', 'yolov8'):  # newer / older SAHI naming
            try:
                return AutoDetectionModel.from_pretrained(
                    model_type=mt, model_path=args.weights,
                    confidence_threshold=conf, device=sahi_device)
            except Exception as e:  # noqa: BLE001
                last = e
        raise last

    sahi_model = make_sahi(0.001)
    results = {}

    # ---- accuracy passes (conf=0.001) ----
    for sz in args.baseline_imgsz:
        name = f'baseline_imgsz{sz}'
        print(f'\n[{name}] predicting...')
        preds = predict_baseline(model, images, sz, 0.001, args.device)
        r = coco_eval(gt, preds)
        r.update(precision_recall_at(gt, preds, conf=0.25))
        results[name] = r
        print(f'  mAP50={r["mAP50"]:.2f}  mAP50-95={r["mAP50_95"]:.2f}  '
              f'P={r["precision"]*100:.1f}  R={r["recall"]*100:.1f}')

    name = f'sahi_{args.slice}_ov{args.overlap}'
    print(f'\n[{name}] predicting (slow)...')
    preds = predict_sahi(sahi_model, images, args.slice, args.overlap)
    r = coco_eval(gt, preds)
    r.update(precision_recall_at(gt, preds, conf=0.25))
    results[name] = r
    print(f'  mAP50={r["mAP50"]:.2f}  mAP50-95={r["mAP50_95"]:.2f}  '
          f'P={r["precision"]*100:.1f}  R={r["recall"]*100:.1f}')

    # ---- timing pass (conf=0.25, same images for every mode) ----
    timing_imgs = images[:args.time_n]
    sahi_model.confidence_threshold = 0.25
    for sz in args.baseline_imgsz:
        ms = time_ms(lambda p, sz=sz: model.predict(source=str(p), imgsz=sz, conf=0.25,
                                                    device=args.device, verbose=False),
                     timing_imgs)
        results[f'baseline_imgsz{sz}'].update({'latency_ms': ms, 'fps': 1000 / ms})
    from sahi.predict import get_sliced_prediction
    ms = time_ms(lambda p: get_sliced_prediction(
        str(p), sahi_model, slice_height=args.slice, slice_width=args.slice,
        overlap_height_ratio=args.overlap, overlap_width_ratio=args.overlap, verbose=0),
        timing_imgs)
    results[name].update({'latency_ms': ms, 'fps': 1000 / ms})

    meta = {'split': args.split, 'n_images': len(images), 'n_gt_boxes': len(gt['annotations']),
            'gpu': gpu, 'python': platform.python_version(), 'torch': torch.__version__,
            'ultralytics': md.version('ultralytics'), 'sahi': md.version('sahi'),
            'timing_images': len(timing_imgs), 'evaluator': f'pycocotools bbox, maxDets={MAX_DETS}',
            'weights': args.weights}

    # ---- report ----
    print('\n| Config | mAP@50 | mAP@50-95 | Precision | Recall | Latency (ms) | FPS |')
    print('|---|---|---|---|---|---|---|')
    for k, v in results.items():
        print(f'| {k} | {v["mAP50"]:.2f} | {v["mAP50_95"]:.2f} | {v["precision"]*100:.1f} | '
              f'{v["recall"]*100:.1f} | {v["latency_ms"]:.1f} | {v["fps"]:.1f} |')
    print('\nPer-class AP@50:')
    print('| Class | ' + ' | '.join(results) + ' |')
    print('|---|' + '---|' * len(results))
    for c in CLASS_NAMES:
        print(f'| {c} | ' + ' | '.join(f'{v["per_class_AP50"][c]:.1f}' for v in results.values()) + ' |')
    print(f'\nHardware: {gpu}')

    out = args.out or f'results_{args.split}.json'
    Path(out).write_text(json.dumps({'meta': meta, 'results': results}, indent=2))
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
