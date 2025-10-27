import cv2
import numpy as np
import datetime
import argparse
import os
import time
import threading
import smtplib
import csv
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from collections import deque, defaultdict
import json

# ==================== CONFIGURATION ====================
parser = argparse.ArgumentParser(description="Advanced YOLO Object Detection System")
parser.add_argument("--source", type=str, default="webcam", 
                    help="Video source: 'webcam', video file path, or RTSP URL")
parser.add_argument("--roi", action="store_true",
                    help="Enable Region of Interest monitoring")
parser.add_argument("--alert-objects", type=str, default="person,car,truck",
                    help="Comma-separated list of objects to trigger alerts (e.g., 'person,car')")
parser.add_argument("--email", action="store_true",
                    help="Enable email alerts")
parser.add_argument("--record-duration", type=int, default=10,
                    help="Duration of video clip to record (seconds)")
parser.add_argument("--cooldown", type=int, default=300,
                    help="Cooldown between alerts (seconds)")
parser.add_argument("--confidence", type=float, default=0.5,
                    help="Detection confidence threshold (0.0-1.0)")
parser.add_argument("--no-display", action="store_true",
                    help="Run headless (no GUI display)")
args = parser.parse_args()

# ==================== DIRECTORIES SETUP ====================
output_dir = "outputs"
clips_dir = os.path.join(output_dir, "clips")
logs_dir = os.path.join(output_dir, "logs")
screenshots_dir = os.path.join(output_dir, "screenshots")

for directory in [output_dir, clips_dir, logs_dir, screenshots_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"✅ Created directory: {directory}")

# ==================== EMAIL CONFIGURATION ====================
EMAIL_CONFIG = {
    "enabled": args.email,
    "sender": "your_email@gmail.com",  # CHANGE THIS
    "password": "your_app_password",    # CHANGE THIS (use App Password for Gmail)
    "receiver": "receiver_email@gmail.com",  # CHANGE THIS
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587
}

# ==================== ROI ZONES (Multiple zones support) ====================
ROI_ZONES = [
    {"name": "Zone 1", "coords": (400, 100, 250, 350), "color": (255, 200, 100)},
    # Add more zones if needed:
    # {"name": "Zone 2", "coords": (100, 100, 200, 200), "color": (100, 255, 200)},
]

# ==================== GLOBAL VARIABLES ====================
alert_objects = [obj.strip().lower() for obj in args.alert_objects.split(",")]
last_alert_time = 0
alert_cooldown = args.cooldown
recording = False
video_writer = None
frame_buffer = deque(maxlen=150)  # Buffer for pre-event recording (5 sec @ 30fps)
detection_history = defaultdict(list)
total_detections = defaultdict(int)

# ==================== CSV LOGGING ====================
log_file = os.path.join(logs_dir, f"detections_{datetime.datetime.now().strftime('%Y%m%d')}.csv")
csv_lock = threading.Lock()

def init_csv_log():
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Object', 'Confidence', 'In_ROI', 'Zone', 'Alert_Triggered'])

def log_detection(obj_label, confidence, in_roi, zone_name, alert_triggered):
    with csv_lock:
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            writer.writerow([timestamp, obj_label, f"{confidence:.2f}", in_roi, zone_name, alert_triggered])

# ==================== EMAIL ALERT FUNCTION ====================
def send_email_alert(video_path, detected_objects, zone_name):
    if not EMAIL_CONFIG["enabled"]:
        return
    
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_CONFIG["sender"]
        msg['To'] = EMAIL_CONFIG["receiver"]
        msg['Subject'] = f"🚨 ALERT: {', '.join(detected_objects)} detected in {zone_name}"
        
        body = f"""
        <html>
        <body>
            <h2>Security Alert</h2>
            <p><strong>Time:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Location:</strong> {zone_name}</p>
            <p><strong>Detected Objects:</strong> {', '.join(detected_objects)}</p>
            <p><strong>Action Required:</strong> Review attached video clip</p>
            <hr>
            <p style="color: gray; font-size: 12px;">This is an automated alert from your YOLO Detection System</p>
        </body>
        </html>
        """
        msg.attach(MIMEText(body, 'html'))
        
        # Attach video file
        if os.path.exists(video_path):
            with open(video_path, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(video_path)}")
                msg.attach(part)
        
        # Send email
        server = smtplib.SMTP(EMAIL_CONFIG["smtp_server"], EMAIL_CONFIG["smtp_port"])
        server.starttls()
        server.login(EMAIL_CONFIG["sender"], EMAIL_CONFIG["password"])
        server.send_message(msg)
        server.quit()
        
        print(f"✅ Email alert sent successfully to {EMAIL_CONFIG['receiver']}")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

# ==================== VIDEO RECORDING FUNCTIONS ====================
def start_recording(width, height):
    global recording, video_writer
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    video_path = os.path.join(clips_dir, f"alert_{timestamp}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))
    
    # Write buffered frames (pre-event)
    for buffered_frame in frame_buffer:
        video_writer.write(buffered_frame)
    
    recording = True
    print(f"🎥 Started recording: {video_path}")
    return video_path

def stop_recording():
    global recording, video_writer
    if video_writer:
        video_writer.release()
        recording = False
        print("⏹️  Recording stopped")

# ==================== MODEL LOADING ====================
print("🔄 Loading YOLOv3 model...")
net = cv2.dnn.readNet("weights/yolov3.weights", "cfg/yolov3.cfg")
# For faster processing, uncomment these lines:
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
print("✅ Model loaded successfully")

# ==================== VIDEO SOURCE ====================
if args.source == "webcam":
    cap = cv2.VideoCapture(0)
    print("📹 Starting webcam feed...")
else:
    cap = cv2.VideoCapture(args.source)
    print(f"📹 Processing video: {args.source}")

if not cap.isOpened():
    print("❌ Error: Could not open video source")
    exit()

# Initialize CSV logging
init_csv_log()

# ==================== MAIN DETECTION LOOP ====================
font = cv2.FONT_HERSHEY_PLAIN
frame_count = 0
fps_start_time = time.time()
fps = 0
recording_start_time = None
current_video_path = None
alert_triggered_objects = set()

print("\n🚀 System started! Press 'q' to quit, 's' to screenshot\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Video stream ended")
        break
    
    frame_count += 1
    height, width, channels = frame.shape
    
    # Add frame to buffer
    frame_buffer.append(frame.copy())
    
    # Calculate FPS
    if frame_count % 30 == 0:
        fps = 30 / (time.time() - fps_start_time)
        fps_start_time = time.time()
    
    # Draw ROI zones
    roi_display_frame = frame.copy()
    if args.roi:
        for zone in ROI_ZONES:
            x, y, w, h = zone["coords"]
            overlay = roi_display_frame.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), zone["color"], -1)
            roi_display_frame = cv2.addWeighted(overlay, 0.3, roi_display_frame, 0.7, 0)
            cv2.rectangle(roi_display_frame, (x, y), (x + w, y + h), zone["color"], 2)
            cv2.putText(roi_display_frame, zone["name"], (x, y-10), font, 1.5, zone["color"], 2)
    
    frame = roi_display_frame
    
    # YOLO Detection
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > args.confidence:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, args.confidence, 0.4)
    
    object_counts = {}
    roi_detections = {}
    current_frame_alert_objects = set()
    
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            
            center_x, center_y = x + w // 2, y + h // 2
            in_roi = False
            zone_name = "Outside"
            
            # Check if object is in any ROI zone
            if args.roi:
                for zone in ROI_ZONES:
                    rx, ry, rw, rh = zone["coords"]
                    if rx < center_x < rx + rw and ry < center_y < ry + rh:
                        in_roi = True
                        zone_name = zone["name"]
                        color = (0, 0, 255)  # Red for ROI detections
                        
                        # Count objects in ROI
                        if label not in roi_detections:
                            roi_detections[label] = 0
                        roi_detections[label] += 1
                        
                        # Check if this object should trigger alert
                        if label.lower() in alert_objects:
                            current_frame_alert_objects.add(label)
                        break
            
            # Count all detections
            if label not in object_counts:
                object_counts[label] = 0
            object_counts[label] += 1
            total_detections[label] += 1
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label_text = f"{label} {int(confidence * 100)}%"
            cv2.putText(frame, label_text, (x, y - 5), font, 1.5, color, 2)
            
            # Log detection
            log_detection(label, confidence, in_roi, zone_name, False)
    
    # Alert logic
    current_time = time.time()
    if (current_frame_alert_objects and 
        args.roi and 
        (current_time - last_alert_time) > alert_cooldown):
        
        if not recording:
            current_video_path = start_recording(width, height)
            recording_start_time = current_time
            alert_triggered_objects = current_frame_alert_objects.copy()
            last_alert_time = current_time
    
    # Handle ongoing recording
    if recording:
        video_writer.write(frame)
        elapsed = current_time - recording_start_time
        
        if elapsed >= args.record_duration:
            stop_recording()
            
            # Send email in background thread
            if EMAIL_CONFIG["enabled"] and current_video_path:
                zone_detected = "ROI Zone"
                threading.Thread(
                    target=send_email_alert,
                    args=(current_video_path, list(alert_triggered_objects), zone_detected),
                    daemon=True
                ).start()
            
            alert_triggered_objects.clear()
    
    # Display information panel
    info_y = 30
    cv2.putText(frame, f"FPS: {int(fps)}", (10, info_y), font, 2, (0, 255, 0), 2)
    info_y += 35
    
    if args.roi:
        cv2.putText(frame, f"ROI Detections:", (10, info_y), font, 2, (255, 255, 255), 2)
        info_y += 30
        for obj, count in roi_detections.items():
            text = f"  {obj}: {count}"
            color = (0, 0, 255) if obj.lower() in alert_objects else (0, 255, 0)
            cv2.putText(frame, text, (10, info_y), font, 1.8, color, 2)
            info_y += 25
    else:
        cv2.putText(frame, f"Total Detections:", (10, info_y), font, 2, (255, 255, 255), 2)
        info_y += 30
        for obj, count in object_counts.items():
            cv2.putText(frame, f"  {obj}: {count}", (10, info_y), font, 1.8, (0, 255, 0), 2)
            info_y += 25
    
    # Recording indicator
    if recording:
        cv2.circle(frame, (width - 30, 30), 10, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (width - 80, 40), font, 2, (0, 0, 255), 2)
    
    # Display frame
    if not args.no_display:
        cv2.imshow("Advanced YOLO Detection System", frame)
    
    # Key controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\n🛑 Shutting down...")
        break
    elif key == ord('s'):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        screenshot_path = os.path.join(screenshots_dir, f"screenshot_{timestamp}.jpg")
        cv2.imwrite(screenshot_path, frame)
        print(f"📸 Screenshot saved: {screenshot_path}")

# Cleanup
if recording:
    stop_recording()

cap.release()
cv2.destroyAllWindows()

# Print summary
print("\n" + "="*50)
print("📊 SESSION SUMMARY")
print("="*50)
print(f"Total Frames Processed: {frame_count}")
print(f"Detection Log: {log_file}")
for obj, count in total_detections.items():
    print(f"  {obj}: {count}")
print("="*50)
print("✅ System shutdown complete")