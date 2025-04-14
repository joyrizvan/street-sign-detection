# 🛑 Street Sign Detection System

This project performs real-time **street sign detection**, **OCR**, and **text-to-speech (TTS)** using:
- YOLOv8 (`ultralytics`)
- OpenCV (`cv2`)
- PyTorch
- PyTesseract for OCR
- pyttsx3 for TTS

> Developed to run inside a Conda environment on **WSL2** or **Linux**.

---

## 📦 Features

- Detect street signs using YOLOv8
- Extract text from signs using Tesseract OCR
- Read detected text aloud using TTS (pyttsx3)
- Designed for performance and modularity

---

## 🧪 Requirements

- Python 3.8+
- Conda / Miniconda
- Linux/WSL2 (tested)
- Webcam access (optional, not supported directly in WSL2)

---

## ⚙️ Setup Instructions

### 1️⃣ Create the environment

```bash
conda create -n img-processing python=3.8 -y
conda activate img-processing
```
### 2️⃣ Install Python dependencies

```bash
conda create -n img-processing python=3.8 -y
conda activate img-processing
```

### 3️⃣ Install system packages (Linux / WSL2 only)
```bash
sudo apt update
sudo apt install -y libopenh264-6 espeak-ng tesseract-ocr
```

If cv2 throws an OpenH264 error, run:
```bash
ln -s $CONDA_PREFIX/lib/libopenh264.so.6 $CONDA_PREFIX/lib/libopenh264.so.5
```

### 🐍 Python Dependencies
Listed in requirements.txt:
```bash
opencv-python
torch
pyttsx3
pytesseract
numpy
ultralytics
```


### 🚀 Running the App
Make sure your img-processing environment is activated.
```bash
python sign_detector.py
```

