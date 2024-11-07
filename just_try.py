import cv2
import numpy as np
import time
import os
count_original=0
count_plotted=0

def people_count():

    # Loading the YOLO model
    yolo_weights = "/home/pi/project/yolov3/yolov3.weights"  
    yolo_config = "/home/pi/project/yolov3/yolov3.cfg"
    coco_names = "/home/pi/project/yolov3/coco.names" 

    net = cv2.dnn.readNet(yolo_weights, yolo_config)
    
    global count_original
    global count_plotted

    # Load the class names for YOLO (COCO dataset contains "person" class)
    with open(coco_names, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    #index for the "person" class
    person_index = classes.index("person")

    # NMS parameters
    nms_threshold = 0.4
    confidence_threshold = 0.5
    
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open video device")
        return None
#     while True:
    ret, frame = cap.read()
    # Check if frame is captured successfully
    if not ret or frame is None:
        print("Error: Could not read frame")
        return None
    while True:
        image_path="/home/pi/project/image/original_image/"
        count_original+=1
        cv2.imwrite(f"{image_path}image{count_original}.png", frame)  # Save your processed image
        break

    # Get the frames dimensions
    (h, w) = frame.shape[:2]

    # Prepare the input blob for the YOLO model
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

    # Set the blob as the input for the YOLO model
    net.setInput(blob)

    # Get the YOLO output layer names
    output_layers = net.getUnconnectedOutLayersNames()

    # Perform forward propagation to get detections
    detections = net.forward(output_layers)

    # Lists to store boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    # Loop through YOLO detections to extract bounding boxes
    for detection in detections[0]:
        scores = detection[5:]  # Class scores
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if class_id == person_index and confidence > confidence_threshold:  # Confidence threshold
            # Calculate the bounding box coordinates
            center_x, center_y, width, height = (detection[:4] * np.array([w, h, w, h])).astype(int)
            startX, startY = int(center_x - width / 2), int(center_y - height / 2)
            endX, endY = int(center_x + width / 2), int(center_y + height / 2)

            # Add bounding box to the list
            boxes.append([startX, startY, width, height])
            confidences.append(confidence)

    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    
    print(indices)

    # Count of valid detections
    people_count = len(indices)

    # Display the number of people detected in the terminal
    print("People detected:", people_count)
    
    # Draw rectangles around detected people
    if people_count > 0:
        for i in indices:
            box = boxes[i]
            startX, startY, width, height = box
            cv2.rectangle(frame, (startX, startY), (startX + width, startY + height), (0, 255, 0), 2)
            #To save image
            image_path="/home/pi/project/image/Plotted_image/"
            count_plotted+=1
            cv2.imwrite(f"{image_path}image{count_plotted}.png", frame)  # Save your processed image
            break
    elif people_count == 0:
        image_path="/home/pi/project/image/Plotted_image/"
        count_plotted+=1
        cv2.imwrite(f"{image_path}image{count_plotted}.png", frame)  # Save your processed image

        
    return people_count
        # Exit on 'q' key press
#         if cv2.waitKey(1000) & 0xFF == ord('q'):
#             break

    cap.release()
    cv2.destroyAllWindows()

