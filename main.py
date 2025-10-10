import cv2
import numpy as np
import datetime
import argparse
import os

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="YOLO Object Detection")
parser.add_argument("--source", type=str, default="webcam",
                    help="Source of the video feed (e.g., 'webcam', 'uk-street.mp4')")
args = parser.parse_args()

# --- Paths ---
OUTPUT_DIR = r"E:\SEM 5\AI\YOLO-Real-Time-Object-Detection\outputs"

# Create the output folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load YOLOv3 Model ---
print("✅ Using full YOLOv3 model")
net = cv2.dnn.readNet("weights/yolov3.weights", "cfg/yolov3.cfg")

# Load class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# --- Load Video Source ---
if args.source == "webcam":
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(args.source)

font = cv2.FONT_HERSHEY_PLAIN

# --- Main Detection Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Detect objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Analyze detections
    class_ids = []
    confidences = []
    boxes = []

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
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 20), font, 2, color, 2)

    # Display output
    cv2.imshow("YOLO Object Detection", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = f"capture_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
        filepath = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(filepath, frame)
        print(f"✅ Screenshot saved at: {filepath}")

cap.release()
cv2.destroyAllWindows()
