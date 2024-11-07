import cv2
import numpy as np
from ultralytics import YOLO
import os
import requests
import time

# Initialize global variables for counting
count_original = 0
count_plotted = 0
people_count_history = []  # Store history of detected people counts

# Function to send data to ThingSpeak
def send_to_thingspeak(current_people_count):
    api_key = 'F6EQ31NYV76I9CRK'  # Replace with your ThingSpeak channel API key
    base_url = f'https://api.thingspeak.com/update?api_key={api_key}&field1='+str(current_people_count)
    
    # Send the data to ThingSpeak
    response = requests.get(base_url)
    if response.status_code == 200:
        print("Data sent to ThingSpeak successfully.")
    else:
        print("Failed to send data to ThingSpeak.")

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

    while True:  # Continuous loop for capturing frames
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
        current_people_count = len(results[0].boxes)  # The boxes attribute contains detected objects
        
        # Display the number of people detected in the terminal
        print("People detected:", current_people_count)
        
        # Send the current people count to ThingSpeak
        time.sleep(1)
        send_to_thingspeak(current_people_count)
        
        # Save the annotated frame
        image_path = "/home/pi/project/image/plotted/"
        count_plotted += 1
        os.makedirs(image_path, exist_ok=True)
        cv2.imwrite(f"{image_path}image{count_plotted}.png", annotated_frame)
        
        # Display the annotated frame in a window (optional)
        #cv2.imshow("Annotated Frame", annotated_frame)

    cap.release()
    cv2.destroyAllWindows()
    
# Example usage
# while True:
if __name__=="__main__":
    people_count()
    #time.sleep(1)  # Adjust the sleep time as needed
