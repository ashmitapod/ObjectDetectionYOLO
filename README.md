# 🧠 YOLO Real-Time Object Detection

This project uses the **YOLOv3** algorithm to perform real-time object detection on video files and live webcam feeds. It's built in **Python** using **OpenCV** and pre-trained YOLO weights.

---

## 🚀 Features

* **Real-Time Detection:** Identifies objects in live webcam feeds.
* **Video File Processing:** Can analyze and detect objects in pre-recorded video files.
* **Optional Region of Interest (ROI):** Activate a specific detection zone — only objects inside the ROI will be counted and highlighted.
* **Dynamic Object Counting:** Displays a live count of all detected objects (or just those in the ROI).
* **Save Screenshots:** Press the `'s'` key during detection to save the current labeled frame to the `outputs` folder.

---

## 🧩 Requirements

* Python 3.6+
* OpenCV
* NumPy

Install the necessary Python libraries:

```bash
pip install numpy opencv-python
```

---

## 📦 Project Setup

### 1️⃣ Clone the Repository

Run these commands from your terminal to clone the project:

```bash
git clone https://github.com/himanshuuyadav/YOLO-Object-Detection.git
cd YOLO-Real-Time-Object-Detection
```

### 2️⃣ Create Required Folders

```bash
mkdir weights cfg outputs
```

### 3️⃣ Download YOLOv3 and YOLOv3-Tiny Resources

Run these commands from your main project directory (e.g., `YOLO-Real-Time-Object-Detection`).

#### Download YOLOv3 weights (236 MB)

```bash
curl -L -o weights/yolov3.weights "https://pjreddie.com/media/files/yolov3.weights"
```

#### Download YOLOv3 config

```bash
curl -L -o cfg/yolov3.cfg "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
```

#### Download YOLOv3-Tiny weights (34 MB)

```bash
curl -L -o weights/yolov3-tiny.weights "https://pjreddie.com/media/files/yolov3-tiny.weights"
```

#### Download YOLOv3-Tiny config

```bash
curl -L -o cfg/yolov3-tiny.cfg "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg"
```

---

## ⚙️ How to Use

All functionality is handled by the `main.py` script. You can specify the video source and enable ROI using command-line arguments.

### ▶️ Use Webcam (Standard Mode)

Detects and counts all objects in the full frame.

```bash
python main.py --source webcam
```

If you don't provide a source, it will default to the webcam.

### 🎯 Use Webcam (ROI Mode)

Activates the detection zone — only objects inside the blue ROI box are highlighted in red and counted.

```bash
python main.py --source webcam --roi
```

### 📹 Process a Video File

You can use the `--roi` flag with video files as well.

**Standard mode:**

```bash
python main.py --source usa-street.mp4
```

**ROI mode:**

```bash
python main.py --source usa-street.mp4 --roi
```

### 💾 Save a Frame

While the detection window is open:

* Press `'s'` to save a screenshot → stored automatically in the `outputs` folder.
* Press `'q'` to quit.

---

## 📁 Folder Structure

```
YOLO-Real-Time-Object-Detection/
│
├── cfg/                  # YOLO configuration files
├── weights/              # Pre-trained YOLO weights
├── outputs/              # Saved screenshots
├── main.py               # Main detection script
├── coco.names            # COCO class names file
└── README.md             # Documentation
```

---

## 🧠 Notes


* You can swap YOLOv3 and YOLOv3-Tiny in the code if you prefer faster detection (with lower accuracy).

---


