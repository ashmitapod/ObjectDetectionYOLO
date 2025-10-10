import cv2
import numpy as np
import datetime
import argparse
import os # Import the os module

# --- 1. ADD A NEW --roi FLAG ---
parser = argparse.ArgumentParser(description="YOLO Object Detection with optional ROI")
parser.add_argument("--source", type=str, default="webcam", 
                    help="Source of the video feed (e.g., 'webcam', 'usa-street.mp4')")
parser.add_argument("--roi", action="store_true",
                    help="Enable Region of Interest (ROI) for detection and counting.")
args = parser.parse_args()

# --- ROI DEFINITION (only used if --roi is enabled) ---
roi_x, roi_y, roi_w, roi_h = 400, 100, 250, 350
alert_color = (0, 0, 255) # Red for alerts
roi_color = (255, 200, 100) # Light blue for the ROI box

# --- SETUP OUTPUT DIRECTORY ---
output_dir = "outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"✅ Created directory: {output_dir}")

# --- Model Loading ---
print("✅ Loading full YOLOv3 model...")
net = cv2.dnn.readNet("weights/yolov3.weights", "cfg/yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# --- Video Source Loading ---
if args.source == "webcam":
    cap = cv2.VideoCapture(0)
    print("✅ Starting webcam feed...")
else:
    cap = cv2.VideoCapture(args.source)
    print(f"✅ Processing video file: {args.source}")

font = cv2.FONT_HERSHEY_PLAIN
while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ End of video file reached.")
        break
        
    height, width, channels = frame.shape
    
    # --- 2. DRAW THE ROI ONLY IF THE FLAG IS ENABLED ---
    if args.roi:
        overlay = frame.copy()
        cv2.rectangle(overlay, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), roi_color, -1)
        alpha = 0.3
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), roi_color, 2)

    object_counts = {}
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]] # Default color

            # --- 3. COUNTING AND ALERT LOGIC DEPENDS ON ROI FLAG ---
            if args.roi:
                center_x, center_y = x + w // 2, y + h // 2
                is_inside_roi = (roi_x < center_x < roi_x + roi_w and 
                                 roi_y < center_y < roi_y + roi_h)
                
                if is_inside_roi:
                    color = alert_color # Change color to red if inside ROI
                    object_counts[label] = object_counts.get(label, 0) + 1
            else:
                # If ROI is off, count every detected object
                object_counts[label] = object_counts.get(label, 0) + 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 20), font, 2, color, 2)

    # --- 4. DISPLAY THE CORRECT COUNT TITLE ---
    y_offset = 40
    count_title = "In ROI:" if args.roi else "Detections:"
    cv2.putText(frame, count_title, (10, y_offset), font, 2.5, (0,0,0), 3)
    y_offset += 40
    for label, count in object_counts.items():
        count_text = f"- {label.capitalize()}: {count}"
        cv2.putText(frame, count_text, (10, y_offset), font, 2, (0, 255, 0), 2)
        y_offset += 30

    cv2.imshow("YOLO Object Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("✅ Exiting program.")
        break
    elif key == ord('s'):
        # Create a unique filename with a timestamp
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"capture_{timestamp}.jpg"
        # Join the directory path and filename
        save_path = os.path.join(output_dir, filename)
        # Save the current frame to the specified path
        cv2.imwrite(save_path, frame)
        print(f"✅ Screenshot saved as {save_path}")

cap.release()
cv2.destroyAllWindows()

