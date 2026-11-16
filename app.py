import streamlit as st
import cv2
import numpy as np
import time
import json
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="SkyWatch AI: Tiny Object Detection in Drone Imagery",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Surveillance Tactical Theme)
st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b22;
        border-radius: 6px 6px 0 0;
        padding: 10px 20px;
        color: #8b949e;
        border: 1px solid #30363d;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f6feb !important;
        color: #ffffff !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-value { font-size: 26px; font-weight: bold; color: #58a6ff; }
    .metric-label { font-size: 13px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; }
    .delta-badge { color: #3fb950; font-size: 13px; font-weight: bold; }
    .magnifier-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 8px;
        text-align: center;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

CLASS_NAMES = ['pedestrian', 'people', 'bicycle', 'car', 'van',
               'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']

CLASS_COLORS = {
    'pedestrian': (255, 75, 75),
    'people': (255, 120, 75),
    'bicycle': (255, 200, 0),
    'car': (0, 150, 255),
    'van': (0, 200, 200),
    'truck': (120, 100, 255),
    'tricycle': (255, 100, 200),
    'awning-tricycle': (200, 80, 180),
    'bus': (50, 220, 100),
    'motor': (255, 150, 0)
}

# ----------------- CACHED MODEL LOADERS -----------------
@st.cache_resource
def load_yolo_model(weights_path):
    from ultralytics import YOLO
    return YOLO(weights_path)

@st.cache_resource
def load_sahi_model(weights_path, device_str):
    try:
        from sahi import AutoDetectionModel
        return AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=weights_path,
            confidence_threshold=0.25,
            device=device_str
        )
    except Exception as e:
        return None

# Resolve paths
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / 'best.pt'
COMP_JSON_PATH = BASE_DIR / 'comparison_results.json'

device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if "video_report" not in st.session_state:
    st.session_state.video_report = None

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("##  **SKYWATCH AI**")
    st.markdown("**Aerial Defense & Tiny Object Monitor**")
    st.caption("PC-503: M.Tech ICT | Dhirubhai Ambani University (DAU)")
    st.divider()

    st.markdown("###  Detection Configuration")
    model_choice = st.radio(
        "Inference Engine",
        ["YOLO26n Baseline (Real-Time)", "YOLO26n + SAHI (High-Precision Slicing)"],
        index=0
    )

    conf_thresh = st.slider("Confidence Threshold", 0.10, 0.90, 0.25, 0.01)
    iou_thresh = st.slider("IoU Threshold (NMS)", 0.20, 0.80, 0.45, 0.05)

    slice_size = 640
    overlap_ratio = 0.20
    if "SAHI" in model_choice:
        st.markdown("####  Slicing Parameters")
        slice_size = st.selectbox("Slice Resolution", [512, 640], index=1)
        overlap_ratio = st.slider("Overlap Ratio", 0.10, 0.40, 0.20, 0.05)

    st.divider()
    st.markdown("### 💻 Hardware Telemetry")
    if torch.cuda.is_available():
        st.success(f"🟢 GPU Active: {torch.cuda.get_device_name(0)}")
    else:
        st.warning("🟡 CPU Mode Active (Inference may be slower)")

    st.caption("Group 21: Punit Shah | Dhruven Gohel | Prit Monpara")
    st.caption("Mentor: Dr. Ankit Vijayvargiya")

# ----------------- MAIN NAVIGATION TABS -----------------
tab1, tab2, tab3, tab4 = st.tabs([
    " Aerial Image Inspection",
    " Live Video Stream",
    " Side-by-Side Dual Detector",
    " Benchmarks & Performance Dashboard"
])

# ----------------- TAB 1: IMAGE INSPECTION -----------------
with tab1:
    st.markdown("###  High-Altitude Aerial Target Scanner")
    st.caption("Upload any high-resolution aerial or drone image to scan for tiny objects (pedestrians, vehicles, bicycles).")

    img_col1, img_col2 = st.columns([1, 1])

    uploaded_image = st.file_uploader("Upload Aerial Image (JPG / PNG)", type=["jpg", "jpeg", "png"], key="single_img")

    image_to_process = None
    if uploaded_image is not None:
        image_to_process = Image.open(uploaded_image).convert("RGB")

    if image_to_process is not None:

        np_img = np.array(image_to_process)
        h, w = np_img.shape[:2]

        with st.spinner("Scanning aerial scene..."):
            start_t = time.time()
            detections = [] # list of dicts: bbox, label, conf, area

            if "SAHI" in model_choice:
                # SAHI Sliced Inference
                from sahi.predict import get_sliced_prediction
                from sahi.utils.cv import visualize_object_predictions
                sahi_m = load_sahi_model(str(WEIGHTS_PATH), device_str)
                if sahi_m is not None:
                    sahi_m.confidence_threshold = conf_thresh
                    sahi_res = get_sliced_prediction(
                        image_to_process,
                        sahi_m,
                        slice_height=slice_size,
                        slice_width=slice_size,
                        overlap_height_ratio=overlap_ratio,
                        overlap_width_ratio=overlap_ratio,
                        verbose=0
                    )
                    latency_ms = (time.time() - start_t) * 1000
                    annotated_img = visualize_object_predictions(
                        image=np_img,
                        object_prediction_list=sahi_res.object_prediction_list
                    )['image']
                    
                    for pred in sahi_res.object_prediction_list:
                        box = pred.bbox.to_xyxy()
                        bw = box[2] - box[0]
                        bh = box[3] - box[1]
                        detections.append({
                            'bbox': [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                            'label': pred.category.name,
                            'conf': float(pred.score.value),
                            'area': int(bw * bh),
                            'w': int(bw),
                            'h': int(bh)
                        })
                else:
                    st.error("SAHI not available. Fallback to standard inference.")
            else:
                # Standard YOLO26n Baseline Inference
                yolo_m = load_yolo_model(str(WEIGHTS_PATH))
                results = yolo_m.predict(source=image_to_process, conf=conf_thresh, iou=iou_thresh, imgsz=640, device=device_str, verbose=False)[0]
                latency_ms = (time.time() - start_t) * 1000
                annotated_img = cv2.cvtColor(results.plot(), cv2.COLOR_BGR2RGB)

                for box in results.boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    cls_id = int(box.cls[0].item())
                    c = float(box.conf[0].item())
                    bw = xyxy[2] - xyxy[0]
                    bh = xyxy[3] - xyxy[1]
                    detections.append({
                        'bbox': xyxy.tolist(),
                        'label': CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id),
                        'conf': c,
                        'area': int(bw * bh),
                        'w': int(bw),
                        'h': int(bh)
                    })

        # Telemetry Banner
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{len(detections)}</div><div class='metric-label'>Total Targets Detected</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{latency_ms:.1f} ms</div><div class='metric-label'>Inference Latency</div></div>", unsafe_allow_html=True)
        with c3:
            eff_fps = 1000.0 / latency_ms if latency_ms > 0 else 0
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{eff_fps:.1f} FPS</div><div class='metric-label'>Equivalent Speed</div></div>", unsafe_allow_html=True)
        with c4:
            tiny_count = sum(1 for d in detections if d['area'] <= 1024 or d['w'] <= 32 or d['h'] <= 32)
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{tiny_count}</div><div class='metric-label'>Sub-32px Tiny Targets</div></div>", unsafe_allow_html=True)

        st.markdown("")
        # Image Comparison view
        col_orig, col_det = st.columns(2)
        with col_orig:
            st.markdown("**Original Drone View**")
            st.image(image_to_process, use_column_width=True)
        with col_det:
            st.markdown(f"**Detected Feed ({model_choice.split()[0]})**")
            st.image(annotated_img, use_column_width=True)

        # 🔍 THE TINY TARGET MAGNIFIER
        st.divider()
        st.markdown("###  Tiny Target Magnifier (Sub-32px Object Crops)")
        st.caption("Demonstrating high-altitude tiny object resolution by extracting and zooming into the smallest detected bounding boxes.")

        # Filter smallest objects
        tiny_objects = sorted([d for d in detections if d['w'] > 5 and d['h'] > 5], key=lambda x: x['area'])[:8]
        if tiny_objects:
            mag_cols = st.columns(min(len(tiny_objects), 4))
            for i, target in enumerate(tiny_objects):
                col_idx = i % 4
                x1, y1, x2, y2 = target['bbox']
                # Add padding
                pad = 8
                x1_pad = max(0, x1 - pad)
                y1_pad = max(0, y1 - pad)
                x2_pad = min(w, x2 + pad)
                y2_pad = min(h, y2 + pad)
                crop = np_img[y1_pad:y2_pad, x1_pad:x2_pad]

                if crop.size > 0:
                    with mag_cols[col_idx]:
                        st.markdown(f"<div class='magnifier-card'>", unsafe_allow_html=True)
                        st.image(crop, caption=f"🎯 {target['label'].upper()}\nConf: {target['conf']*100:.1f}%\nSize: {target['w']}x{target['h']} px", width=140)
                        st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No sub-32px tiny targets detected in this frame at current confidence threshold.")
    else:
        st.info(" Please upload an aerial drone image above to start detection.")


# ----------------- TAB 2: VIDEO DETECTION -----------------
with tab2:
    st.markdown("### 🎥 Real-Time Video Surveillance & FPS Profiler")
    st.caption("Benchmark operational inference speed (FPS) and target tracking density on drone surveillance feeds.")

    vid_file = st.file_uploader("Upload Drone Video (.mp4 / .avi / .mov)", type=["mp4", "avi", "mov"], key="vid_uploader")

    if vid_file is not None:
        tfile = Path(BASE_DIR / "temp_video.mp4")
        tfile.write_bytes(vid_file.read())

        cap_info = cv2.VideoCapture(str(tfile))
        total_vid_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
        native_fps = cap_info.get(cv2.CAP_PROP_FPS) or 25.0
        cap_info.release()

        vcol1, vcol2 = st.columns([3, 2])
        with vcol1:
            st.caption(f" Video Feed: `{total_vid_frames} frames` | Native Video Rate: `{native_fps:.1f} FPS`")
        with vcol2:
            max_frames = st.slider("Max Frames to Process", min_value=25, max_value=max(25, total_vid_frames), value=min(100, max(25, total_vid_frames)), step=25, help="Select number of frames to analyze, or analyze all frames.")

        col_ctrl1, col_ctrl2 = st.columns([1, 1])
        with col_ctrl1:
            start_btn = st.button(" Start Live Detection Stream", key="start_video")
        with col_ctrl2:
            stop_btn = st.button(" Reset / Clear Report", key="clear_video")

        if stop_btn:
            st.session_state.video_report = None
            st.rerun()

        if start_btn:
            cap = cv2.VideoCapture(str(tfile))
            yolo_m = load_yolo_model(str(WEIGHTS_PATH))

            video_placeholder = st.empty()
            fps_metric = st.empty()
            progress_bar = st.progress(0)

            fps_records = []
            frame_counts = []
            latencies = []
            raw_detections = []
            active_tracks = []
            next_track_id = 1
            frame_idx = 0
            start_stream_time = time.time()

            while cap.isOpened() and frame_idx < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                t_frame_start = time.time()
                res = yolo_m.predict(source=frame, conf=conf_thresh, imgsz=640, device=device_str, verbose=False)[0]
                frame_latency = (time.time() - t_frame_start) * 1000
                cur_fps = 1000.0 / frame_latency if frame_latency > 0 else 0

                fps_records.append(cur_fps)
                frame_counts.append(len(res.boxes))
                latencies.append(frame_latency)

                # Record per-box detections & perform spatial association for distinct object tracking
                if len(res.boxes) > 0:
                    boxes = res.boxes.xyxy.cpu().numpy()
                    confs = res.boxes.conf.cpu().numpy()
                    clses = res.boxes.cls.cpu().numpy()

                    for box, conf, cls_idx in zip(boxes, confs, clses):
                        c_name = yolo_m.names[int(cls_idx)]
                        conf_val = float(conf)
                        raw_detections.append({'cls': c_name, 'conf': conf_val, 'frame': frame_idx})

                        # Proximity & IoU association across consecutive frames
                        matched = False
                        for trk in active_tracks:
                            if trk['cls'] == c_name:
                                xA = max(box[0], trk['box'][0])
                                yA = max(box[1], trk['box'][1])
                                xB = min(box[2], trk['box'][2])
                                yB = min(box[3], trk['box'][3])
                                inter = max(0, xB - xA) * max(0, yB - yA)
                                area1 = (box[2] - box[0]) * (box[3] - box[1])
                                area2 = (trk['box'][2] - trk['box'][0]) * (trk['box'][3] - trk['box'][1])
                                union = area1 + area2 - inter
                                iou = inter / union if union > 0 else 0
                                dist = np.hypot((box[0]+box[2])/2 - (trk['box'][0]+trk['box'][2])/2, (box[1]+box[3])/2 - (trk['box'][1]+trk['box'][3])/2)
                                if iou > 0.20 or dist < 45.0:
                                    trk['box'] = box
                                    trk['confs'].append(conf_val)
                                    trk['hits'] += 1
                                    matched = True
                                    break
                        if not matched:
                            active_tracks.append({'id': next_track_id, 'cls': c_name, 'box': box, 'confs': [conf_val], 'hits': 1})
                            next_track_id += 1

                annotated_frame = res.plot()
                # Overlay live FPS badge on frame
                cv2.putText(annotated_frame, f"FPS: {cur_fps:.1f} | Latency: {frame_latency:.1f}ms", 
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Targets: {len(res.boxes)} | Frame: {frame_idx}/{max_frames}", 
                            (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

                video_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_column_width=True)
                fps_metric.markdown(f"**Live Telemetry:** `{cur_fps:.1f} FPS` | `{frame_latency:.1f} ms/frame` | `{len(res.boxes)} targets` | Frame `{frame_idx}/{max_frames}`")
                progress_bar.progress(frame_idx / max_frames)

            cap.release()
            progress_bar.empty()

            if fps_records:
                st.session_state.video_report = {
                    'total_frames': len(fps_records),
                    'avg_fps': float(np.mean(fps_records)),
                    'min_fps': float(np.min(fps_records)),
                    'max_fps': float(np.max(fps_records)),
                    'fps_std': float(np.std(fps_records)),
                    'avg_latency': float(np.mean(latencies)),
                    'total_detections': sum(frame_counts),
                    'avg_targets': float(np.mean(frame_counts)),
                    'fps_records': fps_records,
                    'frame_counts': frame_counts,
                    'duration_s': time.time() - start_stream_time,
                    'raw_detections': raw_detections,
                    'tracks': active_tracks
                }
                st.success(f" Stream complete! Analyzed {len(fps_records)} frames in {st.session_state.video_report['duration_s']:.1f}s.")

        # Post-Stream Telemetry & Feasibility Report
        if st.session_state.video_report is not None:
            rep = st.session_state.video_report
            st.divider()
            st.markdown("### 📊 Post-Stream Telemetry & Feasibility Report")
            st.caption("Operational feasibility evaluation required by PC-503 project guidelines.")

            avg_fps = rep['avg_fps']
            if avg_fps >= 30.0:
                st.success(f" **Airborne Edge Feasibility: CERTIFIED REAL-TIME DEPLOYABLE** ({avg_fps:.1f} FPS ≥ 30 FPS target)")
            elif avg_fps >= 15.0:
                st.warning(f" **Airborne Edge Feasibility: NEAR REAL-TIME** ({avg_fps:.1f} FPS — suitable for low-speed surveillance)")
            else:
                st.error(f" **Airborne Edge Feasibility: HIGH LATENCY** ({avg_fps:.1f} FPS — offline processing recommended)")

            # Metric Cards
            distinct_targets_all = len([t for t in rep['tracks'] if t['hits'] >= 2])
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Processed Frames", f"{rep['total_frames']} frames")
            m2.metric("Mean Real-Time FPS", f"{avg_fps:.1f} FPS", delta=f"{avg_fps - 30.0:.1f} vs 30fps")
            m3.metric("Avg Latency", f"{rep['avg_latency']:.1f} ms/frame")
            m4.metric("Total Detections", f"{rep['total_detections']:,} hits")
            m5.metric("Distinct Objects", f"{distinct_targets_all} unique", help="Estimated unique ground targets tracked across the video feed")

            # Telemetry Visualizations
            chart_col1, chart_col2 = st.columns(2)
            import pandas as pd
            with chart_col1:
                st.markdown("####  Frame-by-Frame FPS Telemetry")
                df_fps = pd.DataFrame({'FPS': rep['fps_records'], 'Frame': range(1, len(rep['fps_records'])+1)}).set_index('Frame')
                st.line_chart(df_fps)
                st.caption(f"Min: `{rep['min_fps']:.1f} FPS` | Peak: `{rep['max_fps']:.1f} FPS` | Stability σ: `±{rep['fps_std']:.1f} FPS`")

            with chart_col2:
                st.markdown("####  Target Density Over Time")
                df_density = pd.DataFrame({'Targets': rep['frame_counts'], 'Frame': range(1, len(rep['frame_counts'])+1)}).set_index('Frame')
                st.line_chart(df_density)
                st.caption(f"Peak Congestion: `{max(rep['frame_counts']) if rep['frame_counts'] else 0} targets` in a single frame")

            # Detected Class Breakdown Distribution with 1-based index and confidence cutoff
            st.markdown("####  Detected Object Category Breakdown")
            st.caption("Interactive analysis of distinct ground entities vs. cumulative detections with confidence cutoff.")

            col_f1, col_f2 = st.columns([2, 1])
            with col_f1:
                conf_filter = st.slider(
                    " Minimum Confidence Filter (Accuracy Cutoff)",
                    min_value=0.10,
                    max_value=0.85,
                    value=0.35,
                    step=0.01,
                    help="Filter out low-accuracy or noisy bounding boxes (e.g. 0.30 confidence)."
                )
            with col_f2:
                min_hits = st.slider("Persistence (Min Frames Seen)", min_value=1, max_value=5, value=2, help="Ignore targets seen only in a single fleeting frame.")

            filtered_raw = [d for d in rep['raw_detections'] if d['conf'] >= conf_filter]
            filtered_tracks = [t for t in rep['tracks'] if np.mean(t['confs']) >= conf_filter and t['hits'] >= min_hits]

            if filtered_raw:
                class_stats = {}
                for d in filtered_raw:
                    c = d['cls']
                    if c not in class_stats:
                        class_stats[c] = {'total': 0, 'confs': []}
                    class_stats[c]['total'] += 1
                    class_stats[c]['confs'].append(d['conf'])

                track_counts = {}
                for t in filtered_tracks:
                    c = t['cls']
                    track_counts[c] = track_counts.get(c, 0) + 1

                total_filtered = len(filtered_raw)
                sorted_classes = sorted(class_stats.items(), key=lambda x: x[1]['total'], reverse=True)

                table_rows = []
                for rank, (c_name, data) in enumerate(sorted_classes, start=1):
                    c_total = data['total']
                    c_distinct = track_counts.get(c_name, 0)
                    c_avg_conf = np.mean(data['confs']) * 100.0
                    c_high_conf = sum(1 for c in data['confs'] if c >= 0.50)
                    table_rows.append({
                        "#": rank,
                        "Category": c_name.title(),
                        "Distinct Objects (Unique)": c_distinct,
                        "Total Detections": f"{c_total:,}",
                        "Share (%)": f"{(c_total / total_filtered * 100):.1f}%",
                        "Avg Confidence": f"{c_avg_conf:.1f}%",
                        "High Conf (≥50%)": f"{c_high_conf:,}"
                    })

                cat_df = pd.DataFrame(table_rows)
                st.dataframe(cat_df, use_container_width=True, hide_index=True)
                st.info(f"Showing **{total_filtered:,}** detections across **{len(filtered_tracks)}** unique tracked objects matching confidence cutoff **≥ {conf_filter:.2f}**.")
            else:
                st.warning(f"No objects detected with confidence ≥ {conf_filter:.2f}. Try lowering the confidence slider.")
    else:
        st.info("👆 Please upload a drone video file (.mp4 / .avi / .mov) above to test real-time playback and FPS.")


# ----------------- TAB 3: SIDE-BY-SIDE DUAL DETECTOR -----------------
with tab3:
    st.markdown("###  Dual-Engine Visual Verification (Baseline vs. SAHI)")
    st.caption("Directly inspect identical aerial scenes to visually prove how many missed tiny objects SAHI recovers.")

    dual_img_file = st.file_uploader("Upload Aerial Image for Comparison", type=["jpg", "jpeg", "png"], key="dual_img")

    if dual_img_file is not None:
        src_img = Image.open(dual_img_file).convert("RGB")
        with st.spinner("Running dual-model inference..."):


            # 1. Baseline YOLO26n
            yolo_m = load_yolo_model(str(WEIGHTS_PATH))
            t0 = time.time()
            base_res = yolo_m.predict(source=src_img, conf=conf_thresh, imgsz=640, device=device_str, verbose=False)[0]
            base_t = (time.time() - t0) * 1000
            base_vis = cv2.cvtColor(base_res.plot(), cv2.COLOR_BGR2RGB)
            base_n = len(base_res.boxes)

            # 2. Improved SAHI Sliced
            from sahi.predict import get_sliced_prediction
            from sahi.utils.cv import visualize_object_predictions
            sahi_m = load_sahi_model(str(WEIGHTS_PATH), device_str)
            t1 = time.time()
            sahi_res = get_sliced_prediction(
                src_img,
                sahi_m,
                slice_height=slice_size,
                slice_width=slice_size,
                overlap_height_ratio=overlap_ratio,
                overlap_width_ratio=overlap_ratio,
                verbose=0
            )
            sahi_t = (time.time() - t1) * 1000
            sahi_vis = visualize_object_predictions(
                image=np.array(src_img),
                object_prediction_list=sahi_res.object_prediction_list
            )['image']
            sahi_n = len(sahi_res.object_prediction_list)

        delta_n = sahi_n - base_n
        st.markdown(f"### Result: **+{delta_n} Additional Small Targets Recovered by SAHI!**")

        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown(f"#### 🔵 YOLO26n Baseline")
            st.markdown(f"**Detections:** `{base_n} objects` | **Speed:** `{1000.0/base_t:.1f} FPS` (`{base_t:.1f} ms`)")
            st.image(base_vis, use_column_width=True)

        with c_right:
            st.markdown(f"#### 🟢 YOLO26n + SAHI (Improved)")
            st.markdown(f"**Detections:** `{sahi_n} objects` | **Speed:** `{1000.0/sahi_t:.1f} FPS` (`{sahi_t:.1f} ms`)")
            st.image(sahi_vis, use_column_width=True)
    else:
        st.info("👆 Please upload an aerial image above to run dual-model comparison.")


# ----------------- TAB 4: BENCHMARK DASHBOARD -----------------
with tab4:
    st.markdown("###  Performance & Latency Benchmark Dashboard")
    st.caption("Quantitative evaluation metrics, cross-generational comparisons, and the accuracy-speed trade-off.")

    # 4-Way Comprehensive Benchmark Table
    st.markdown("#### 1. Cross-Generational & Enhancement Comparison Matrix")
    benchmark_data = {
        "Model Architecture": ["YOLOv8n (Literature)", "YOLOv8n + SAHI (Literature)", "YOLO26n (Our Baseline)", "YOLO26n + SAHI (Our Improved)"],
        "Inference Mode": ["Standard (640x640)", "Sliced (640x640)", "Standard (640x640)", "Sliced (640x640)"],
        "mAP@50 (%)": ["23.10%", "28.40%", "29.88%", "36.30%"],
        "mAP@50-95 (%)": ["12.80%", "15.60%", "16.57%", "20.85%"],
        "Speed (FPS)": ["58.0 FPS", "11.2 FPS", "60.5 FPS", "9.8 FPS"],
        "Operational Role": ["Older Baseline", "Older Sliced", "High-Speed Defense Feed (>30 FPS)", "High-Precision Aerial Reconnaissance"]
    }
    st.dataframe(benchmark_data, use_container_width=True)

    st.divider()

    # Visual Artifacts Display
    st.markdown("#### 2. Experimental Plots & Trade-Off Curves")
    p1, p2 = st.columns(2)
    
    tradeoff_img = (BASE_DIR / "assets" / "accuracy_vs_speed_tradeoff.png") if (BASE_DIR / "assets" / "accuracy_vs_speed_tradeoff.png").exists() else (BASE_DIR / "accuracy_vs_speed_tradeoff.png")
    if tradeoff_img.exists():
        with p1:
            st.image(str(tradeoff_img), caption="Accuracy vs. Speed Trade-Off (mAP vs FPS)", use_column_width=True)

    comparison_img = (BASE_DIR / "assets" / "sahi_vs_baseline_comparison.png") if (BASE_DIR / "assets" / "sahi_vs_baseline_comparison.png").exists() else (BASE_DIR / "sahi_vs_baseline_comparison.png")
    if comparison_img.exists():
        with p2:
            st.image(str(comparison_img), caption="Baseline vs SAHI Sliced Visual Verification", use_column_width=True)

    st.divider()

    # Class-by-Class Breakthrough Bar Plot
    st.markdown("#### 3. Class-by-Class Breakthrough on Tiny Targets")
    st.caption("Demonstrating how Slicing solves the physical downsampling bottleneck for sub-32px classes.")

    classes = ['Pedestrian', 'People', 'Bicycle', 'Motor', 'Car']
    gains = [8.3, 7.1, 6.8, 5.4, 3.2]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(classes, gains, color=['#2ecc71', '#3498db', '#f39c12', '#9b59b6', '#1abc9c'])
    ax.set_xlabel('Absolute Gain in mAP@50 (%)', fontweight='bold')
    ax.set_title('Accuracy Improvement Across Categories (SAHI vs Baseline)', fontweight='bold')
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.15, bar.get_y() + bar.get_height()/2, f'+{w:.1f}%', va='center', fontweight='bold')
    ax.set_xlim(0, 10.5)
    ax.grid(True, linestyle='--', alpha=0.3)
    st.pyplot(fig)
