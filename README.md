# 🧠 AI Real-Time Emotion, Health (rPPG) & Focus Tracker

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ONNX DirectML](https://img.shields.io/badge/Hardware%20Accel-DirectML%20%7C%20CUDA-green.svg)](https://onnxruntime.ai/)
[![Privacy: 100% Local](https://img.shields.io/badge/Privacy-100%25%20On--Device-brightgreen.svg)]()

A lightweight, research-grade, multimodal computer vision and affective computing system that runs **100% locally** using a single standard RGB webcam. 

It real-time tracks **contactless physiological vitals (rPPG heart rate)**, **AffectNet deep learning emotions**, **3D head pose & iris eye-gaze focus**, and **ISO PERCLOS fatigue**, complete with **local SQLite logging**, a **Windows system tray daemon**, and an **Ollama-powered Local AI Health Coach**.

---

## 🌟 Key Features

- **❤️ Contactless Physiology (rPPG & HRV)**: Measures blood volume pulse (BVP) via facial micro-color shifts (POS/CHROM algorithms), Butterworth bandpass filtering, and FFT power spectrum to calculate real-time **Heart Rate (BPM)** and pulse waveforms without wearable sensors.
- **😊 Deep Learning Emotion Recognition**: Powered by an **AffectNet-pretrained HSEmotion ONNX** model with **DirectML GPU acceleration**, temporal exponential moving average (EMA) smoothing, and desk neutral-prior calibration to eliminate resting-face false positives.
- **👀 3D Iris Eye-Gaze & Attention Fusion**: Uses MediaPipe 468 3D mesh + refined iris points (468–477) fused with SolvePnP 3D head pose (Pitch/Yaw/Roll) for robust screen attention & focus scoring ($0–100\%$).
- **😴 ISO PERCLOS Drowsiness & Fatigue Detection**: Tracks percentage of eye closure over rolling 30-second windows and blink rates (blinks/min) to prevent momentary blinks from triggering false alarms.
- **⚡ Multimodal Stress Index (MSI: 0–100)**: Fuses autonomic pulse elevation ($40\%$), blink dynamics ($25\%$), facial tension ($20\%$), and head restlessness ($15\%$) into a unified stress score with non-blocking desktop popups and guided breathing exercises.
- **🤖 Local LLM AI Health Coach (Ollama / Llama-3 / Gemma-2)**: Ingests daily session analytics and generates personalized, empathetic ergonomics & wellness summaries with zero cloud API keys.
- **🟢 Windows System Tray App & Smart Inactivity Standby**:
  - Runs in the background at **$1.0\text{ Hz}$ sampling cadence** ($< 2-4\%$ CPU utilization).
  - Automatically yields the camera when other apps (Zoom, Teams, OBS) open and reacquires it seamlessly.
  - Automatically stops, organizes logs into JSON/CSV, and enters deep $0\%$ CPU standby after 15 minutes of absence, waking up instantly when you move your mouse.

---

## 🏗️ System Architecture

```
                               ┌─────────────────────────────┐
                               │  Single Local Webcam Stream │
                               └──────────────┬──────────────┘
                                              │
                                              ▼
                               ┌─────────────────────────────┐
                               │ MediaPipe 468 3D Mesh + Iris│
                               └──────┬───────┬───────┬──────┘
                                      │       │       │
             ┌────────────────────────┘       │       └────────────────────────┐
             ▼                                ▼                                ▼
  ┌──────────────────────┐        ┌──────────────────────┐        ┌──────────────────────┐
  │  1. Face & Gaze      │        │  2. AffectNet Model  │        │  3. rPPG Physiology  │
  ├──────────────────────┤        ├──────────────────────┤        ├──────────────────────┤
  │• 3D Head Pose (P/Y/R)│        │• 8 Emotion Classes   │        │• Forehead POS rPPG   │
  │• Iris Pupil Vector   │        │• Temporal EMA Filter │        │• Heart Rate (BPM)    │
  │• Screen Focus Score  │        │• Valence / Arousal   │        │• ISO PERCLOS Fatigue │
  └──────────┬───────────┘        └──────────┬───────────┘        └──────────┬───────────┘
             │                               │                               │
             └────────────────────────┬──────┴───────────────────────────────┘
                                      ▼
             ┌───────────────────────────────────────────────────────────────┐
             │      Multimodal Stress Index & Real-Time Logger (1 Hz)        │
             ├───────────────────────────────────────────────────────────────┤
             │• SQLite Database (`tracker_history.db`)                       │
             │• Non-blocking Windows Desktop Alerts                          │
             │• System Tray Controller (`tray_app.py`)                       │
             │• Daily Local LLM Health Coach (`ai_coach.py`)                 │
             └───────────────────────────────────────────────────────────────┘
```

---

## 📋 System & Hardware Requirements

### Minimum Hardware
- **Processor**: Intel Core i3 / AMD Ryzen 3 or higher (ARM64 supported)
- **RAM**: 4 GB RAM minimum (8 GB recommended)
- **Camera**: Any standard built-in or USB webcam ($720\text{p}$ or $1080\text{p}$, $30\text{ FPS}$)
- **GPU Acceleration**: Compatible with all DirectX 12 GPUs (Intel Iris Xe, AMD Radeon, NVIDIA GeForce) via **DirectML**

### Operating System
- **Windows 10 / 11** (Full support for System Tray, DirectShow camera sharing, and Windows input wake-up)
- **macOS / Linux** (Full CLI & visual tracker support)

### Core Python Dependencies (`requirements.txt`)
| Package | Version | Purpose |
| :--- | :--- | :--- |
| **`opencv-python`** | `~4.11.0` | Camera frame acquisition & HUD rendering |
| **`mediapipe`** | `==0.10.14` | 468 3D Facial Mesh & Refined Iris Landmarks |
| **`numpy`** | `1.26.x` | High-speed array tensor & matrix calculations |
| **`scipy`** | `>=1.10.0` | Butterworth bandpass filter & FFT signal processing |
| **`onnxruntime-directml`** | `>=1.18.0` | GPU-accelerated DirectML deep learning inference |
| **`hsemotion-onnx`** | `>=0.3.1` | AffectNet EfficientNet facial emotion recognition |
| **`pystray`** | `>=0.19.5` | Windows System Tray notification area controller |
| **`pandas`** | `>=1.5.0` | Session log analytics and CSV/JSON data export |
| **`Pillow`** | `>=10.0.0` | Dynamic system tray icon image generator |
| **`requests`** | `>=2.28.0` | Local Ollama LLM REST API client |

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 installed
- (Optional for deep LLM synthesis) [Ollama](https://ollama.com) with `llama3` or `gemma2`

### 2. Installation
Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/ai-emotion-health-face-tracker.git
cd ai-emotion-health-face-tracker
pip install -r requirements.txt
```

### 3. Usage

#### Option A: Interactive Visual HUD Window
```bash
python main.py
```
- Press **`Q`** inside the camera window to save and exit.
- Press **`E`** to export the live session directly to CSV.

#### Option B: Silent Background System Tray App
```bash
python tray_app.py
```
- Minimizes directly into the Windows Notification Area tray.
- Right-click the **green circle icon** to pause, ask the AI Coach, or view reports.

#### Option C: Generate Daily AI Health Coach Insights
```bash
python run_coach.py
```

---

## 📊 Database & Log Structure

All data is stored strictly on your local machine in `tracker_history.db`:
- **`sessions`**: Start/end time, duration, average heart rate, dominant emotion, and total alerts.
- **`metrics`**: 1-Hz time-series logging (BPM, emotion, EAR, MAR, head pose, focus score, stress level).
- **`events`**: Timestamped camera states, high-stress popups, and drowsiness warnings.
- **`session_reports/`**: Formatted JSON analytical breakdowns and CSV spreadsheets generated at the end of each session.

---

## 🛠️ Tech Stack & Dependencies

- **Computer Vision**: `OpenCV`, `MediaPipe` (468 FaceMesh + Iris)
- **Deep Learning / Accelerators**: `ONNX Runtime`, `DirectML`, `HSEmotion` (AffectNet)
- **Signal Processing**: `SciPy` (Butterworth Bandpass, FFT), `NumPy`
- **Data & UI**: `SQLite3`, `Pandas`, `Pystray`, `Pillow`
- **Local AI**: `Ollama API` (Llama-3, Gemma-2, Mistral)

---

## 🔒 Security Architecture & Privacy Disclosures

This repository is designed following **Privacy-by-Design** and **Local-Only Zero-Trust** principles:

### 1. Zero Cloud Transmission & No Video Recording
- **No Video Storage**: Raw camera frames exist only in transient RAM buffers for immediate landmark calculation ($< 30\text{ ms}$) and are immediately discarded. No video files, images, or audio recordings are ever saved to disk.
- **Zero Telemetry / No External Calls**: All neural network inferences (`MediaPipe`, `AffectNet ONNX`), signal filters, database logging, and LLM coaching queries run entirely on `localhost`. No data is transmitted across the network.

### 2. Open-Source Transparency & Anti-Malware Notice
- **100% Plain Python Source**: All code is open, auditable, and uses standard, official PyPI packages (`opencv-python`, `mediapipe`, `onnxruntime-directml`, `scipy`).
- **No Obfuscated Executables**: No pre-compiled binary blobs (`.exe` or `.dll`) are included in the source tree to ensure verifiable safety.
- **Local SQLite Storage**: Health metrics are stored in a standard unencrypted SQLite database (`tracker_history.db`) in your local project directory. Users can delete or inspect this file at any time.

---

## ⚕️ Medical & Wellness Disclaimer

This software is an **experimental wellness and productivity tracking tool**. It is **NOT** a medical diagnostic device, clinical software, or certified medical monitor. Measurements such as rPPG heart rate, eye strain, and stress scores are approximations derived from computer vision algorithms and should **not** be used as a substitute for professional medical advice, diagnosis, or treatment.

---

## 📜 License
MIT License. Feel free to use, modify, and contribute!
