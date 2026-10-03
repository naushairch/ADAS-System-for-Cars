# 🚗 Advisory ADAS — Android / Python / TFLite / UFLD

> **Smart Real-time Driver Assistance & Safety System for Mobile Devices**

An advanced, real-time forward collision warning (FCW), lane departure warning (LDW), and speed limit voice advisory system built for Android (Java / TFLite) and Python (PyTorch / OpenCV). Mounted on a vehicle dashboard, it uses phone camera vision to detect hazardous traffic conditions and issue low-latency voice alerts.

---

## 🌟 Key Features & Implemented Modules

| Feature Module | Tech Stack & Models | Implementation Status |
|---|---|---|
| **Real-time Camera Stream** | CameraX with backpressure management | ✅ **Done** |
| **Object Detection (FCW)** | YOLOv8n TFLite (Vehicles, Pedestrians, Bicycles) | ✅ **Done** |
| **Ultra-Fast Lane Detection (LDW)** | UFLD (Ultra-Fast-Lane-Detection) / SCNN & TFLite | ✅ **Done** |
| **Lane Departure Warning** | Lateral drift offset estimation & curve geometry | ✅ **Done** |
| **Multi-Object Tracking** | IoU Tracker with persistent Track IDs | ✅ **Done** |
| **Monocular Distance Estimation** | Bounding box geometry + Ego-corridor gating | ✅ **Done** |
| **Time-To-Collision (TTC)** | Smoothed TTC calculation & forward crash prediction | ✅ **Done** |
| **Speed Limit Advisory** | GPS speed tracking vs. Road speed limits | ✅ **Done** |
| **Low-Latency Audio Engine** | SoundPool with bilingual voice alerts (EN / UR) | ✅ **Done** |
| **Session Logging & Analytics** | CSV telemetry logger (Frames, TTC, Alerts, GPS) | ✅ **Done** |
| **CARLA Simulator Integration** | Python live recorder, replay runner & test suite | ✅ **Done** |

---

## 🏗️ Architecture Overview

```
[ CameraX Stream / Video Frame ]
              │
              ├──► [ YOLOv8 Detector ] ──► [ IoU Tracker ] ──► [ Distance & TTC Estimator ]
              │                                                        │
              ├──► [ UFLD Lane Detector ] ──► [ Lane Departure Logic ] ─┤
              │                                                        ▼
              └──► [ GPS Speed Provider ] ──────────────► [ Alert Engine Decision State Machine ]
                                                                       │
                                                                       ▼
                                                       [ Low-Latency Audio Alert (SoundPool) ]
```

---

## 🚀 Getting Started — Setup Guide

### 1. Requirements & Android Setup
* **IDE**: Android Studio Ladybug (or newer) on Windows/Linux/macOS.
* **Target Hardware**: Snapdragon 865 or equivalent (e.g., OnePlus 8T).
* **Package Name**: `com.adas`.
* Import project directly or replace `app/` files in an empty Android Studio Java project.

### 2. Device Preparation
1. Enable **Developer Options** and **USB Debugging** on your Android device.
2. Verify connection via ADB:
   ```bash
   adb devices
   ```

### 3. Model Assets Placement
Place the TFLite models in `app/src/main/assets/`:
* `yolov8n_float32.tflite` — Object Detection
* `ufld_lane_float16.tflite` — Lane Detection

If exporting from PyTorch:
```bash
pip install ultralytics torch
yolo export model=yolov8n.pt format=tflite imgsz=640
python tools/export_lane_tflite.py
```

### 4. Audio Advisory Setup
Generate bilingual audio alerts (English & Urdu):
```bash
pip install gTTS pydub numpy
python tools/generate_alerts.py
```
Place WAV files (`chime.wav`, `en_brake.wav`, `ur_brake.wav`, `en_lane.wav`, `ur_lane.wav`, etc.) into `app/src/main/res/raw/`.

---

## 🎯 System Calibration & Tuning

To ensure millimeter-accurate distance estimation and reliable lane offset calculation:

1. **Mount Phone**: Secure phone in landscape mode near the center dashboard.
2. **Camera Calibration**: Run static target tests at 10m, 20m, and 30m distance markers.
3. **Focal Length Adjustment**: Compute `focal = (boxWidthPx * distanceM) / 1.75` and update `DistanceEstimator.FOCAL_PX`.
4. **Lane Threshold Tuning**: Use `python tools/tune_lane_threshold.py` for lane boundary alignment.

---

## 📊 Analytics & Log Retrieval

Pull telemetry logs directly from the device:
```bash
adb pull /sdcard/Android/data/com.adas/files/Documents/ .
```

Analyse trip metrics with Python tools:
```bash
python tools/trip_summary.py --log_file trip_history.json
python tools/evaluate.py
```

---

## 🛠️ Performance Cheat Sheet

All decision thresholds are tuned in `AlertEngine.java`:

| Issue | Recommended Action |
|---|---|
| **Too many crash alerts** | Increase `CONFIRM_FRAMES`, lower `TTC_WARN_SET` |
| **Delayed crash alerts** | Increase `TTC_URGENT_SET` |
| **Chattering alerts** | Expand gap between `_SET` and `_CLEAR` thresholds |
| **Nuisance alerts on curves** | Adjust `corridorHalfM` or check UFLD lane geometry |
| **Twitchy speed warnings** | Increase `SPEED_CONFIRM_FRAMES` |

---

## 📜 Disclaimer
This system is an **Advisory ADAS** intended for research and supplementary driver assistance only. It never guarantees safety and does not replace active driver attention.
