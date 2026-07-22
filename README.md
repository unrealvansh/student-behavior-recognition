# 🎓 Student Behaviour Recognition System

> **AI-powered real-time classroom monitoring** using Computer Vision & Deep Learning.  
> Presented by Vansh Singhal — K.R. Manglam University (April 2026)

---

## 🧠 The Problem

In a classroom of 40–60 students, **one teacher cannot watch everyone**.  
67% of disengagement happens in blind spots — seats the teacher rarely faces.  
Disengaged students go unnoticed, struggling students don't get help, and manual monitoring leads to teacher burnout.

**Every student deserves attention. Every teacher deserves help.**

---

## 💡 Our Solution: An AI Teaching Assistant

Give the classroom a pair of smart eyes that:
- ✅ Never blink
- ✅ Never get tired
- ✅ Watch every student simultaneously
- ✅ Tell the teacher: *"3 sleeping, 5 on phones, 32 engaged"*

**Real-time. Automatic. Intelligent.**

---

## ⚙️ How It Works: 3-Step Smart Pipeline

```
Raw Video → YOLOv8 → ByteTrack → MobileNetV2 → Dashboard
```

| Step | Model | Purpose | Output |
|------|-------|---------|--------|
| **1. FIND** | YOLOv8m (Ultralytics) | Real-time person detection | Bounding boxes around every student |
| **2. TRACK** | ByteTrack (ECCV 2022) | Persistent ID assignment | Same ID follows each student across frames |
| **3. UNDERSTAND** | MobileNetV2 (Google) | Behavior classification | One of 5 behavior labels per student |

Temporal smoothing eliminates single-frame glitches by voting over 15-frame windows.

---

## 🏷️ 5 Behaviours We Recognise

| # | Behaviour | Meaning | Color | Alert? |
|---|-----------|---------|-------|--------|
| 1 | **Sitting / Standing** | Present & attentive | Gray | No |
| 2 | **Sleeping** | Disengaged | 🔴 Red | Yes |
| 3 | **Using Mobile** | Distracted | 🟡 Yellow | Yes |
| 4 | **Reading / Writing** | Engaged | 🟢 Green | No |
| 5 | **Hand Raising** | Participating | 🔵 Blue | No |

---

## 🛠️ Technology Stack (All Free & Open-Source)

| Layer | Technology |
|-------|-----------|
| **Video Input** | OpenCV |
| **Detector** | YOLOv8 (Ultralytics, 2023) |
| **Tracking** | ByteTrack (ECCV 2022) |
| **CNN Brain** | MobileNetV2 (Google) |
| **Language** | Python 3.10+ |
| **Frameworks** | PyTorch / TorchVision |
| **Dashboard** | Flask + HTML/CSS/JS + SocketIO |

**Total Software Cost: ₹0**

---

## 📁 Project Structure

```
student-behavior-recognition/
├── app.py                      # Flask server (main entry point)
├── src/
│   ├── full_pipeline.py        # 3-Step Smart Pipeline orchestrator
│   ├── tracker.py              # YOLOv8 + ByteTrack person tracker
│   ├── classifier.py           # MobileNetV2 behavior classifier
│   ├── detector.py             # Person detector with sliced inference
│   └── temporal_smoothing.py   # Box-level temporal smoothing
├── models/
│   ├── behavior_classifier.pth # Trained MobileNetV2 weights
│   └── yolo_detector.pt        # Custom YOLO detector (optional)
├── frontend/
│   ├── templates/              # Dashboard HTML templates
│   └── static/                 # Dark theme CSS styling & JS scripts
├── train_classifier.py         # MobileNetV2 training script
├── data/                       # Training dataset (5 behavior classes)
├── tests/                      # Unit tests
└── requirements.txt            # Python dependencies
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Server

```bash
python app.py
```

### 3. Open the Dashboard

Navigate to **http://localhost:5000** in your browser.

### 4. Start Monitoring

Click **"Live Monitoring"** in the sidebar to begin real-time AI detection.

---

## 📋 Requirements

- **Python** 3.10+
- **Hardware**: Intel i5 / 8 GB RAM (no GPU required, but CUDA helps)
- **Webcam**: Any USB or built-in webcam
- **OS**: Windows / Linux / macOS

---

## 🏫 Institution

**K.R. Manglam University**  
Academic Project — April 2026

---

## 📄 License

MIT License