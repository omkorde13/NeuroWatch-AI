# NeuroWatch AI

## Real-Time Neuromorphic Surveillance System

NeuroWatch AI is an advanced real-time surveillance platform that detects violent and abnormal activities using:

- Spiking Neural Networks (SNNs)
- Optical Flow Motion Analysis
- YOLOv8 Human Detection
- FastAPI Backend
- React Frontend Dashboard

The system analyzes live webcam streams, extracts temporal motion patterns, performs spike-based neural inference, and displays live threat analytics through a modern AI dashboard.

---

# Features

## AI Features

- Real-time violence detection
- Optical flow motion analysis
- Temporal spike encoding
- Spiking Neural Network inference
- Human detection using YOLOv8
- Motion threshold filtering

## Frontend Features

- Live webcam feed
- Real-time threat monitoring
- AI analytics dashboard
- Dynamic activity updates
- Modern cyberpunk UI

## Backend Features

- FastAPI server
- Shared AI state engine
- Real-time frontend integration
- Live prediction updates

---

# Tech Stack

## AI / ML

- Python
- PyTorch
- snnTorch
- OpenCV
- YOLOv8
- NumPy

## Backend

- FastAPI
- Uvicorn

## Frontend

- React
- Vite
- TailwindCSS

---

# System Architecture

```text
Webcam Feed
     ↓
YOLO Human Detection
     ↓
Optical Flow Extraction
     ↓
Spike Encoding
     ↓
Spiking Neural Network
     ↓
Violence Prediction
     ↓
FastAPI Backend
     ↓
React Frontend Dashboard
```

---

# Folder Structure

```text
neurowatch-ai/
│
├── ai-engine/
│   ├── snn/
│   └── training/
│
├── backend/
│   └── api/
│
├── frontend/
│
├── datasets/
│
├── models/
│
├── ai_state.json
│
├── requirements.txt
│
└── README.md
```

---

# Installation

## 1. Clone Repository

```bash
git clone <your-repo-url>
cd neurowatch-ai
```

---

## 2. Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

---

## 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

# Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on:

```text
http://localhost:5173
```

---

# Backend Setup

From project root:

```bash
uvicorn backend.api.server:app --reload
```

Backend runs on:

```text
http://127.0.0.1:8000
```

---

# Run Live AI Detection

```bash
python ai-engine/snn/live_real_detection.py
```

This starts:

- webcam capture
- YOLO detection
- optical flow analysis
- SNN inference
- live violence prediction

---

# Model Performance

## Accuracy

```text
76.62%
```

Achieved using:

- temporal motion learning
- optical flow spike encoding
- motion threshold filtering
- real surveillance datasets

---

# Dataset

Used Dataset:

## Real Life Violence Situations Dataset

Contains:

- Violence videos
- NonViolence videos

Used for:

- violence recognition
- surveillance analysis
- abnormal activity detection

---

# AI Pipeline

## 1. Human Detection

YOLOv8 detects people inside frames.

## 2. Optical Flow

Extracts:

- motion magnitude
- acceleration
- motion variance
- direction changes

## 3. Spike Encoding

Motion features are converted into temporal spike sequences.

## 4. Spiking Neural Network

SNN performs real-time violence classification.

---

# Future Improvements

- Multi-camera support
- Cloud deployment
- Transformer-SNN hybrid models
- Edge AI optimization
- Crowd anomaly analysis
- Audio aggression detection
- GPU acceleration

---

# Resume Description

Developed a real-time AI surveillance platform using Spiking Neural Networks (SNNs) for violence detection from live video streams. Built a temporal motion analysis pipeline using optical flow and continuous spike encoding. Integr
