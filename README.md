# 🚁 SkyWatch AI: Tiny Object Detection in Drone Imagery Using YOLO26

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![YOLO26](https://img.shields.io/badge/Ultralytics-YOLO26-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![SAHI](https://img.shields.io/badge/Inference-SAHI%20Slicing-brightgreen.svg)](https://github.com/obss/sahi)
[![Streamlit](https://img.shields.io/badge/GUI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **PC-503: M.Tech (ICT) Project | Dhirubhai Ambani University (DAU)**  
> **Course Instructor:** Dr. Ankit Vijayvargiya  
> **Group 21:** Punit Shah (202611057) · Dhruven Gohel (202611055) · Prit Monpara (202611060)

---

## 📌 1. Project Overview & Defense Relevance

High-altitude drone reconnaissance and defense monitoring systems face a fundamental computer vision bottleneck: **targets such as distant vehicles, high-speed UAVs, and personnel occupy sub-32×32 pixel regions**. In standard deep convolutional networks, repeated strided downsampling (stride 8, 16, 32) destroys the spatial features of tiny objects before reaching the prediction head.

This repository implements an end-to-end aerial surveillance solution benchmarked on the **VisDrone2019-DET** dataset. We:
1. Fine-tuned the latest **YOLO26 Nano** architecture featuring **Small-Target-Aware Label Assignment (STAL)** and **NMS-Free End-to-End** inference.
2. Conducted a deep **convergence and error diagnostic** revealing the physical downsampling limits on wireframe and clustered objects.
3. Integrated **Slicing Aided Hyper Inference (SAHI)** to eliminate downsampling loss on tiny targets.
4. Profiling the **Accuracy vs. Real-Time Video Speed Trade-Off (FPS)** to establish operational surveillance modes.
5. Deployed an interactive **Streamlit Tactical Web GUI** featuring a live video stream, side-by-side dual detector, and a **Tiny Target Magnifier**.

---

## 📊 2. Comprehensive 4-Way Benchmark

| Model Architecture | Inference Mode | mAP@50 (Loose) | mAP@50-95 (Strict) | Speed (FPS) | Latency | Operational Profile |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **YOLOv8n** *(Literature Baseline)* | Standard (640×640) | 23.10% | 12.80% | ~58.0 FPS | 17.2 ms | Older baseline; penalizes sub-32px IoU during matching. |
| **YOLOv8n + SAHI** *(Literature)* | Sliced (640×640) | 28.40% | 15.60% | ~11.2 FPS | 89.3 ms | Slicing recovers small objects, but limited by older backbone. |
| **YOLO26n** *(Our Baseline)* | Standard (640×640) | **29.88%** | **16.57%** | **60.5 FPS** | **16.5 ms** | **High-Speed Real-Time Video Feed (>30 FPS)**. |
| **YOLO26n + SAHI** *(Our Improved)* | Sliced (640×640) | **36.30%** | **20.85%** | **9.8 FPS** | **101.6 ms** | **High-Precision Reconnaissance Scanner**. |

### 🚀 Key Findings:
- **Generational Leap:** Native YOLO26n (`29.88% mAP@50`) outperforms older YOLOv8 + SAHI (`28.40%`) right out of the box due to **STAL**.
- **Tiny-Object Breakthrough:** Combining YOLO26 with SAHI pushes mAP@50 to **36.30% (`+6.42%`)** and strict mAP@50-95 to **20.85% (`+4.28%`)**.
- **Category Gains:**
  - 🚶 **Pedestrian:** `+8.3%`
  - 👥 **People:** `+7.1%`
  - 🚲 **Bicycle:** `+6.8%` *(recovering the exact class that collapsed in baseline)*
  - 🏍️ **Motorcycle:** `+5.4%`
  - 🚗 **Car:** `+3.2%`

---

## 🏗️ 3. Repository Architecture

```text
tiny-object-detection-yolo26/
├── README.md                           <- Project documentation & benchmarks
├── requirements.txt                    <- Python dependencies
├── .gitignore                          <- Git ignore configuration
├── app.py                              <- Interactive Streamlit Tactical Web GUI
├── best.pt                             <- Fine-tuned YOLO26n weights (5.18 MB)
├── comparison_results.json             <- Benchmark data
├── notebooks/
│   └── master_tiny_object_detection.ipynb <- Unified master notebook for all phases
├── configs/
│   └── visdrone.yaml                   <- YOLO dataset configuration
├── demo_samples/
│   ├── visdrone_aerial_sample1.jpg     <- High-resolution aerial test scene
│   ├── visdrone_aerial_sample2.jpg
│   ├── visdrone_aerial_sample3.jpg
│   └── drone_surveillance_feed.mp4     <- Urban traffic surveillance video
└── assets/
    ├── accuracy_vs_speed_tradeoff.png  <- FPS vs Accuracy scatter plot
    └── sahi_vs_baseline_comparison.png <- Visual side-by-side detection proof
```

---

## ⚡ 4. Quickstart Guide

### Option A: Interactive Streamlit Web GUI (Local)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/tiny-object-detection-yolo26.git
   cd tiny-object-detection-yolo26
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the application:**
   ```bash
   streamlit run app.py
   ```
   Open `http://localhost:8501` in your browser.

---

### Option B: Google Colab (Cloud GPU)

1. Open `notebooks/master_tiny_object_detection.ipynb` in [Google Colab](https://colab.research.google.com).
2. Set Runtime to **GPU (T4)**.
3. Run all cells sequentially to reproduce the entire pipeline end-to-end.

---

## 🔬 5. References & Literature

- **Survey Paper:** R. Nautiyal and M. Deshmukh, *"Tiny object detection: An in-depth survey of techniques, challenges, and future directions"*, *Digital Signal Processing*, 2026.
- **VisDrone Benchmark:** AISKYEYE Team, Tianjin University, *"VisDrone: A Benchmark for Vision-Based Drone Tasks"*, IEEE TPAMI.
- **SAHI Framework:** F. C. Akyon et al., *"Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection"*, IEEE ICIP.
- **YOLO26:** Ultralytics Research, 2026.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
