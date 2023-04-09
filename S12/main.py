import os
import cv2
import numpy as np
import urllib.request

# Download YOLOv3 files if they don't already exist
if not os.path.exists("yolov3.weights"):
    urllib.request.urlretrieve("https://pjreddie.com/media/files/yolov3.weights", "yolov3.weights")
if not os.path.exists("yolov3.cfg"):
    urllib.request.urlretrieve("https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg", "yolov3.cfg")

# Load YOLOv3 network
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")


# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0]] for i in net.getUnconnectedOutLayers()]
# Set minimum confidence threshold and class labels
conf_threshold = 0.5
class_labels = []
with open("coco.names", "r") as f:
    class_labels = [line.strip() for line in f.readlines()]

# Load image or video file
input_file = "room_ser.jpeg"
cap = cv2.VideoCapture(input_file)

# Process frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert image to blob format
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

    # Pass blob through network and get predictions
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # Process each output layer
    for output in outputs:
        # Process each detection
        for detection in output:
            # Extract class ID and confidence
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter out weak detections
            if confidence > conf_threshold:
                # Calculate bounding box coordinates
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)

                # Draw bounding box and label on frame
                color = (0, 255, 0)
                cv2.rectangle(frame, (left, top), (left + width, top + height), color, thickness=2)
                label = f"{class_labels[class_id]}: {confidence:.2f}"
                cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=2)

    # Show output frame
    cv2.imshow("YOLOv3", frame)
    if cv2.waitKey(1) == ord("q"):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
