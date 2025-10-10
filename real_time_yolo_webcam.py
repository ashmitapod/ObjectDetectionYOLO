import cv2
import numpy as np
import datetime # <-- 1. IMPORT DATETIME LIBRARY

# Load Yolo
net = cv2.dnn.readNet("weights/yolov3.weights", "cfg/yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Initialize Webcam
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN

while True:
    _, frame = cap.read()
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
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

    cv2.imshow("Webcam Feed", frame)

    # --- 2. ADD THE SAVE LOGIC HERE ---
    key = cv2.waitKey(1)
    if key == ord('q'): # Press 'q' to quit
        break
    elif key == ord('s'): # Press 's' to save the frame
        # Create a unique filename with a timestamp
        filename = f"capture_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
        # Save the current frame to a file
        cv2.imwrite(filename, frame)
        print(f"✅ Screenshot saved as {filename}")

# Release everything
cap.release()
cv2.destroyAllWindows()