import cv2
import numpy as np

# Load YOLOv3 model
net = cv2.dnn.readNet("C:/Users/akash/Documents/data science/DEEP LEARNING/yolo_project/yolov3_custom_final.weights", "C:/Users/akash/Documents/data science/DEEP LEARNING/yolo_project/yolov3_custom.cfg")

with open("C:/Users/akash/Documents/data science/DEEP LEARNING/yolo_project/obj.names.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Capture video from the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Check if frame was captured
    if not ret:
        print("Failed to grab frame")
        break

    height, width, _ = frame.shape

    # Prepare the frame for YOLOv3
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames()
    outs = net.forward(output_layers)

    # Initialize lists for detected bounding boxes, confidences, and class IDs
    boxes, confidences, class_ids = [], [], []

    # Analyze each output from YOLOv3
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

                # Calculate the coordinates for the top-left corner
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression to remove redundant boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels on the frame
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), font, 1, color, 2)

    # Display the resulting frame
    cv2.imshow("Webcam", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
