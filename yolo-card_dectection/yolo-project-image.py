import cv2
import numpy as np

# Load YOLOv3 model
net = cv2.dnn.readNet("C:/Users/akash/Documents/data science/DEEP LEARNING/yolo_project/yolov3_custom_final.weights", "C:/Users/akash/Documents/data science/DEEP LEARNING/yolo_project/yolov3_custom.cfg")

# Load your class labels (if you have a custom .names file)
classes = []
with open("C:/Users/akash/Documents/data science/DEEP LEARNING/yolo_project/obj.names.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Specify the image you want to test
image_path = "C:/Users/akash/Documents/data science/DEEP LEARNING/yolo_project/data/card_demo.jpeg"
img = cv2.imread(image_path)
height, width, _ = img.shape

# Convert the image to a blob
blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)

# Get the output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Perform a forward pass of YOLO
outputs = net.forward(output_layers)

# Extract bounding boxes, confidences, and class IDs
boxes = []
confidences = []
class_ids = []

for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:  # You can adjust this threshold
            # Extract the bounding box coordinates
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply Non-Max Suppression to remove duplicates
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Draw bounding boxes and labels
if len(indices) > 0:
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = (0, 255, 0)  # You can change color if needed
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Show the output image
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally save the output image
cv2.imwrite("output.jpg", img)
