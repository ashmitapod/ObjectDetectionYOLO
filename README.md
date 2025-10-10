# 🧠 YOLO Real-Time Object Detection

This project uses the **YOLOv3** algorithm to perform **real-time object detection** on video files and live webcam feeds.  
It’s built with **Python**, **OpenCV**, and **pre-trained YOLO weights**.

---

## 🚀 Features

-  **Real-Time Detection:** Detects multiple objects live from your webcam.  
- **Video File Processing:** Supports object detection on saved video files.  
- **Command-Line Interface:** Easily switch between webcam and video sources using CLI arguments.  
- **Save Screenshots:** Press **`s`** during detection to save the current frame as an image.  
- **Quit Anytime:** Press **`q`** to exit detection mode.

---

## 🧩 Requirements

- Python **3.6+**
- **OpenCV**
- **NumPy**

Install dependencies:

```bash
pip install numpy opencv-python
```

---

## 📁 Setup

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/YOLO-RealTime-ObjectDetection.git
cd YOLO-RealTime-ObjectDetection
```

2. **Create folders:**

```bash
mkdir weights cfg
```

3. **Download YOLOv3 model files:**

| File Type | File Name | Destination Folder | Download Link |
|------------|------------|--------------------|----------------|
| Weights | `yolov3.weights` | `weights/` | [Download Here](https://pjreddie.com/media/files/yolov3.weights) |
| Config | `yolov3.cfg` | `cfg/` | [Download Here](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg) |
| COCO Labels | `coco.names` | `cfg/` | [Download Here](https://github.com/pjreddie/darknet/blob/master/data/coco.names) |

---

## ▶️ How to Use

All detection functionality is handled by the **`main.py`** script.

### 🟢 Use Your Webcam

```bash
python main.py --source webcam
```

> If no source is provided, the program defaults to the webcam.

### 🎥 Detect Objects in a Video File

```bash
python main.py --source usa-street.mp4
```
or
```bash
python main.py --source uk.mp4
```

### 💾 Save a Frame

While detection is running:

- Press **`s`** → Save the current frame as an image  
- Press **`q`** → Quit the detection window

---

## 🧠 How It Works

1. The YOLOv3 model divides the input image into grids.
2. Each grid predicts bounding boxes and class probabilities.
3. The script uses **OpenCV’s DNN module** to load and run YOLOv3.
4. Detected objects are drawn with bounding boxes and labels in real-time.

---

## 🧱 Project Structure

```
YOLO-RealTime-ObjectDetection/
│
├── main.py
├── cfg/
│   ├── yolov3.cfg
│   └── coco.names
├── weights/
│   └── yolov3.weights
├── outputs/
│   └── (saved screenshots)
└── README.md
```

---

## 📸 Example Output


![alt text](outputs/capture_2025-10-10_11-33-07.jpg)

