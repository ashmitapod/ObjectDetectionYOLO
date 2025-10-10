# 🧠 YOLO Real-Time Object Detection

This project uses the **YOLOv3** algorithm to perform real-time object detection on video files and live webcam feeds. It's built in **Python** using **OpenCV** and pre-trained YOLO weights.

---

## 🚀 Features

* **Real-Time Detection:** Identifies objects in live webcam feeds.
* **Video File Processing:** Can analyze and detect objects in pre-recorded video files.
* **Optional Region of Interest (ROI):** Activate a specific detection zone. Only objects inside the ROI will be counted and highlighted.
* **Dynamic Object Counting:** Displays a live count of all detected objects (or just those in the ROI).
* **Save Screenshots:** Press the `'s'` key during detection to save the current labeled frame to an **outputs/** folder.

---

## 🧩 Requirements

* Python 3.6+
* OpenCV
* NumPy

Install dependencies:

```bash
pip install numpy opencv-python
```

---

## ⚙️ Setup

Run these commands from your **main project directory** (e.g., `YOLO-Real-Time-Object-Detection`).

### Download YOLOv3 weights (236 MB)

```bash
curl -L -o weights/yolov3.weights "https://pjreddie.com/media/files/yolov3.weights"
```

### Download YOLOv3 config

```bash
curl -L -o cfg/yolov3.cfg "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
```

### Download YOLOv3-Tiny weights (34 MB)

```bash
curl -L -o weights/yolov3-tiny.weights "https://pjreddie.com/media/files/yolov3-tiny.weights"
```

### Download YOLOv3-Tiny config

```bash
curl -L -o cfg/yolov3-tiny.cfg "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg"
```

Make sure you have these folders:

```
YOLO-Real-Time-Object-Detection/
│
├── main.py
├── cfg/
│   ├── yolov3.cfg
│   └── yolov3-tiny.cfg
├── weights/
│   ├── yolov3.weights
│   └── yolov3-tiny.weights
├── coco.names
└── outputs/
```

---

## ▶️ How to Use

### 1️⃣ Use your webcam (Standard Mode)

This detects and counts all objects in the full frame.

```bash
python main.py --source webcam
```

### 2️⃣ Use your webcam (ROI Mode)

Add the `--roi` flag to activate the detection zone. Only objects inside the blue box will be counted and highlighted.

```bash
python main.py --source webcam --roi
```

### 3️⃣ Process a video file

You can use the `--roi` flag with video files as well.

**Standard mode:**

```bash
python main.py --source usa-street.mp4
```

**ROI mode:**

```bash
python main.py --source usa-street.mp4 --roi
```

---

## 💾 Saving Frames

While the detection window is open:

* Press **`s`** → Save a screenshot to `outputs/`
* Press **`q`** → Quit the program

Screenshots will be automatically saved as:

```
outputs/capture_YYYY-MM-DD_HH-MM-SS.jpg
```

---

## 📚 Notes

* Make sure `coco.names` is present in the main folder.
* Default detection threshold: 0.5
* Non-maximum suppression threshold: 0.4

---

**Project:** YOLO Real-Time Object Detection
**Built with:** Python, OpenCV, YOLOv3
