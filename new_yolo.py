import cv2
import numpy as np
from ultralytics import YOLO
import os
from app import update_people_count  # Import the function from app.py

count_original = 0
count_plotted = 0

def people_count():
    # Load the YOLOv8 model
    model = YOLO("yolo11x-pose.pt")

    global count_original
    global count_plotted

    # RTSP URL from the Tapo C200
    rtsp_url = 'rtsp://boschcamera:12345678@10.222.74.239/stream1'

    # Initialize the webcam
    cap = cv2.VideoCapture(rtsp_url)
    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open video device")
        return None

    while True:  # Infinite loop to keep the process running
        ret, frame = cap.read()
        # Check if the frame was captured successfully
        if not ret or frame is None:
            print("Error: Could not read frame")
            break

        # Save the original frame
        image_path = "/home/pi/project/image/original/"
        count_original += 1
        os.makedirs(image_path, exist_ok=True)
        cv2.imwrite(f"{image_path}image{count_original}.png", frame)

        # Run YOLOv8 inference on the frame
        results = model.predict(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Count the number of people detected (using YOLOv8's results)
        people_count = len(results[0].boxes)  # The boxes attribute contains detected objects

        # Display the number of people detected in the terminal
        print("People detected:", people_count)

        # Update the people count in the web dashboard
        update_people_count(people_count)

        # Save the annotated frame
        image_path = "/home/pi/project/image/plotted/"
        count_plotted += 1
        os.makedirs(image_path, exist_ok=True)
        cv2.imwrite(f"{image_path}image{count_plotted}.png", annotated_frame)

        # Optional: Limit the frame rate
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

# Example usage
people_count()
