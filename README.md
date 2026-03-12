# Advanced YOLO Object Detection System

A comprehensive real-time object detection system with ROI monitoring, email alerts, video recording, and analytics dashboard.

## Features

### Core Features
-  **Real-time Object Detection** using YOLOv3
-  **Multiple ROI Zones** - Monitor different areas simultaneously
-  **Smart Email Alerts** - Get notified with video clips when specific objects detected
-  **Video Recording** - Automatic clip generation with pre/post event buffer
-  **Multi-source Support** - Webcam, video files, or RTSP streams
-  **Object Counting** - Track detections in real-time
-  **Detection Logging** - CSV logs for data analysis
-  **Analytics Dashboard** - Generate visual reports and insights

### Advanced Features
-  Configurable detection sensitivity
-  Alert cooldown to prevent spam
-  Selective object filtering
-  Pre-event recording buffer
-  FPS counter and performance monitoring
-  Screenshot capture functionality
-  Headless mode for server deployment

---

## Requirements

### System Requirements
- Python 3.7+
- OpenCV 4.x
- 4GB RAM minimum
- Webcam or video source

### Python Dependencies
```bash
pip install opencv-python numpy pandas matplotlib
```

### YOLO Model Files
Download and place in respective folders:

1. **YOLOv3 Weights** (237 MB)
   ```bash
   # Download to weights/ folder
   wget https://pjreddie.com/media/files/yolov3.weights -P weights/
   ```

2. **Configuration files** (Already included)
   - `cfg/yolov3.cfg`
   - `coco.names`

---

## Project Structure

```
project/
├── main.py                    # Main detection script
├── analytics_viewer.py        # Analytics dashboard generator
├── config.json               # Configuration file
├── coco.names               # Object class names
├── weights/
│   └── yolov3.weights       # YOLO weights (download required)
├── cfg/
│   ├── yolov3.cfg          # YOLO configuration
│   └── yolov3-tiny.cfg     # Tiny version (faster)
├── outputs/
│   ├── clips/              # Recorded video clips
│   ├── logs/               # Detection CSV logs
│   └── screenshots/        # Saved screenshots
└── README.md
```

---

## Quick Start

### 1. Basic Detection (Webcam)
```bash
python main.py
```

### 2. With ROI Monitoring
```bash
python main.py --roi
```

### 3. Video File Processing
```bash
python main.py --source usa-street.mp4
```

### 4. Full System with Alerts
```bash
python main.py --roi --email --alert-objects person,car
```

---

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--source` | Video source (webcam/file/RTSP) | webcam |
| `--roi` | Enable ROI monitoring | False |
| `--email` | Enable email alerts | False |
| `--alert-objects` | Objects to trigger alerts | person,car,truck |
| `--confidence` | Detection confidence (0-1) | 0.5 |
| `--record-duration` | Video clip length (seconds) | 10 |
| `--cooldown` | Alert cooldown (seconds) | 300 |
| `--no-display` | Run without GUI | False |

### Examples

**Monitor parking area for vehicles:**
```bash
python main.py --roi --alert-objects car,truck,bus --confidence 0.6
```

**Process recorded video with high sensitivity:**
```bash
python main.py --source traffic.mp4 --confidence 0.4
```

**Server deployment (no display):**
```bash
python main.py --roi --email --no-display
```

---

## Email Setup (Gmail)

### Step 1: Enable 2-Factor Authentication
1. Go to [Google Account Security](https://myaccount.google.com/security)
2. Enable 2-Step Verification

### Step 2: Generate App Password
1. Visit [App Passwords](https://myaccount.google.com/apppasswords)
2. Select "Mail" and "Other (Custom name)"
3. Name it "YOLO Detection"
4. Copy the 16-character password

### Step 3: Update Configuration
Edit `main.py` (lines 40-46):
```python
EMAIL_CONFIG = {
    "enabled": True,  # Change to True
    "sender": "your_email@gmail.com",  # Your Gmail
    "password": "xxxx xxxx xxxx xxxx",  # App password (16 chars)
    "receiver": "alert_email@gmail.com",  # Recipient email
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587
}
```

---

## ROI Configuration

### Editing ROI Zones (main.py, lines 49-53)

```python
ROI_ZONES = [
    {"name": "Entrance", "coords": (400, 100, 250, 350), "color": (255, 200, 100)},
    {"name": "Parking", "coords": (100, 300, 200, 200), "color": (100, 255, 200)},
    {"name": "Exit", "coords": (700, 150, 200, 300), "color": (255, 100, 100)},
]
```

**Format:** `(x, y, width, height)`
- `x, y` = Top-left corner position
- `width, height` = Zone dimensions

**Finding coordinates:**
1. Run: `python main.py --roi`
2. Use screenshot feature (press 's')
3. Open in image editor to measure coordinates

---

## Analytics Dashboard

### Generate Reports
```bash
python analytics_viewer.py
```

### Features
- Detections over time (hourly trends)
- Top detected objects bar chart
- ROI vs non-ROI distribution
- Activity heatmap by hour
- Exportable PNG charts and TXT reports

### Output Files
- `outputs/analytics_report_YYYYMMDD_HHMMSS.png`
- `outputs/summary_report_YYYYMMDD_HHMMSS.txt`

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit program |
| `s` | Save screenshot |

---

## For CSE Students - Project Enhancement Tips

### What Makes This Project Stand Out:

#### 1. **Multi-Feature Integration** 
- Not just detection - full monitoring system
- Email alerts show real-world application
- Logging demonstrates data management skills

#### 2. **Professional Code Quality** 
- Well-structured with comments
- Command-line arguments for flexibility
- Error handling and threading
- Modular design

#### 3. **Analytics Component** 
- Shows data science integration
- Visualization with matplotlib
- Pandas for data processing

#### 4. **Documentation** 
- Complete README
- Clear setup instructions
- Usage examples

### Additional Features to Impress Evaluators:

1. **Create a PowerPoint with:**
   - System architecture diagram
   - Flowcharts of detection logic
   - Screenshots of ROI zones
   - Email alert examples
   - Analytics dashboard images

2. **Prepare Demo Video showing:**
   - Live detection
   - ROI monitoring
   - Email received on phone
   - Analytics generation

3. **Highlight in Presentation:**
   - "Threaded email sending for non-blocking alerts"
   - "Circular buffer for pre-event recording"
   - "CSV logging for future ML training"
   - "Multi-zone ROI for scalability"

---

## Troubleshooting

### Issue: "Could not open video source"
**Solution:** Check webcam permissions or verify video file path

### Issue: Email not sending
**Solutions:**
- Verify Gmail App Password (not regular password)
- Check internet connection
- Enable "Less secure app access" if needed
- Verify SMTP settings

### Issue: Low FPS / Slow detection
**Solutions:**
- Use YOLOv3-tiny for faster processing:
  ```bash
  # Edit main.py line 147 to use tiny model
  net = cv2.dnn.readNet("weights/yolov3-tiny.weights", "cfg/yolov3-tiny.cfg")
  ```
- Reduce frame resolution
- Enable GPU acceleration (requires CUDA)

### Issue: "Module not found"
**Solution:** Install missing packages:
```bash
pip install opencv-python numpy pandas matplotlib
```

---

## Advanced Configuration

### Use GPU Acceleration
Uncomment lines in `main.py`:
```python
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
```
*(Requires OpenCV built with CUDA support)*

### RTSP Camera Support
```bash
python main.py --source "rtsp://username:password@ip:port/stream"
```

### Multiple Alert Recipients
Edit email function to send to multiple addresses:
```python
msg['To'] = ", ".join(["email1@gmail.com", "email2@gmail.com"])
```

---

## Detection Log Format

CSV file structure (`outputs/logs/detections_YYYYMMDD.csv`):

| Column | Description |
|--------|-------------|
| Timestamp | Detection time |
| Object | Detected object class |
| Confidence | Detection confidence (0-1) |
| In_ROI | Boolean (True if in ROI) |
| Zone | ROI zone name |
| Alert_Triggered | Boolean (True if alert sent) |

---

## Video Clip Features

### Automatic Recording Triggers:
- Starts when alert object detected in ROI
- Includes 5 seconds of pre-event footage (buffered)
- Records for specified duration (default 10 sec)
- Saved as `alert_YYYYMMDD_HHMMSS.avi`

### Cooldown System:
- Prevents email spam
- Default: 5 minutes between alerts
- Configurable via `--cooldown` parameter

---

## Additional Resources

### Download YOLOv3 Weights:
```bash
cd weights
wget https://pjreddie.com/media/files/yolov3.weights
```

### Download YOLOv3-Tiny (Faster, smaller):
```bash
wget https://pjreddie.com/media/files/yolov3-tiny.weights
```

### YOLO Documentation:
- Official: https://pjreddie.com/darknet/yolo/
- OpenCV DNN: https://docs.opencv.org/master/d6/d0f/group__dnn.html

---

## Contributors

**CSE Project Team**
- Developed for academic purposes
- Advanced Computer Vision & Machine Learning course
- Real-time monitoring system implementation

---

## License

This project is for educational purposes. YOLO is licensed under the original authors' terms.

---

## Future Enhancements

- [ ] Web dashboard (Flask/Django)
- [ ] Mobile app notifications
- [ ] Person re-identification
- [ ] Facial recognition integration
- [ ] Cloud storage for clips
- [ ] Multiple camera support
- [ ] License plate detection
- [ ] Crowd counting algorithm

---

## Support

For issues or questions:
1. Check Troubleshooting section
2. Review YOLO documentation
3. Verify all dependencies installed
4. Check file paths and permissions

## Quick Commands Cheat Sheet

```bash
# Basic run
python main.py

# With ROI
python main.py --roi

# With email alerts
python main.py --roi --email --alert-objects person

# Process video
python main.py --source video.mp4 --roi

# High sensitivity
python main.py --roi --confidence 0.3

# View analytics
python analytics_viewer.py

# Server mode
python main.py --roi --email --no-display
```
