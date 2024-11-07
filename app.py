import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
import time

# Initialize global variables for counting
count_original = 0
count_plotted = 0
current_people_count = 0

def people_count():
    global count_original
    global count_plotted
    global current_people_count

    # Load the YOLOv8 model
    model = YOLO("yolo11x-pose.pt")

    # RTSP URL from the Tapo C200
    rtsp_url = 'rtsp://boschcamera:12345678@10.222.74.239/stream1'

    # Initialize the webcam
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        st.error("Error: Could not open video device")
        return None
    
    ret, frame = cap.read()
    if not ret or frame is None:
        st.error("Error: Could not read frame")
        cap.release()
        return None

    # Save the original frame
    image_path = "/home/pi/project/image/original/"
    count_original += 1
    os.makedirs(image_path, exist_ok=True)
    cv2.imwrite(f"{image_path}image{count_original}.png", frame)
    
    # Run YOLOv8 inference on the frame
    results = model.predict(frame)
    
    # Count the number of people detected
    current_people_count = len(results[0].boxes)

    # Save the annotated frame
    annotated_frame = results[0].plot()
    image_path = "/home/pi/project/image/plotted/"
    count_plotted += 1
    os.makedirs(image_path, exist_ok=True)
    cv2.imwrite(f"{image_path}image{count_plotted}.png", annotated_frame)

    cap.release()

# Streamlit app interface
st.title("People Detection System")

# Start detection button
if st.button("Start Detection"):
    st.session_state.running = True
    while st.session_state.running:
        people_count()  # Call the people_count function
        st.success(f"People detected: {current_people_count}")
        time.sleep(1)  # Delay for the next detection

# Stop detection button
if st.button("Stop Detection"):
    st.session_state.running = False

# Display the people count in real-time
if 'running' in st.session_state and st.session_state.running:
    st.text(f"Current People Count: {current_people_count}")
