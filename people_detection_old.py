import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import altair as alt
import pandas as pd
import os
import time

# Initialize global variables for counting
count_original = 0
count_plotted = 0
current_people_count = 0

# Initialize session state for tracking the most recent count
if 'latest_count' not in st.session_state:
    st.session_state.latest_count = [0]  # Start with an initial value to avoid empty chart

if 'running' not in st.session_state:
    st.session_state.running = False

# Initialize chart data in session state
if 'chart_data' not in st.session_state:
    st.session_state.chart_data = pd.DataFrame({
        'Index': [],
        'People Count': []
    })

def people_count():
    global count_original
    global count_plotted
    global current_people_count

    # Load the YOLOv8 model
    model = YOLO("yolo11x-pose.pt")
    
    rtsp_url = 'rtsp://boschcamera:12345678@10.222.74.239/stream1'

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

    # Count the number of people detected (class 0 is usually 'person' in COCO dataset)
    current_people_count = sum(1 for box in results[0].boxes if box.cls == 0)

    # Print the number of detected persons to the console
    print(f"People Count: {current_people_count}")  # Log only the person count

    # Save the annotated frame (optional)
    annotated_frame = results[0].plot()
    image_path = "/home/pi/project/image/plotted/"
    count_plotted += 1
    os.makedirs(image_path, exist_ok=True)
    cv2.imwrite(f"{image_path}image{count_plotted}.png", annotated_frame)

    # Update the latest count in the session state
    st.session_state.latest_count.append(current_people_count)
    if len(st.session_state.latest_count) > 6:  # Limit the list size to 6 for better performance
        st.session_state.latest_count.pop(0)

    # Update chart data in session state
    st.session_state.chart_data = pd.DataFrame({
        'Index': list(range(len(st.session_state.latest_count))),
        'People Count': [int(x) for x in st.session_state.latest_count]  # Ensure values are integers
    })

    cap.release()




# Streamlit app interface
st.markdown("<h2 style='text-align: center; color: #2C3E50;'>People Detection System</h2>", unsafe_allow_html=True)

# Create a horizontal container for buttons
button_container = st.container()
with button_container:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        start_button = st.button("Start Detection")
    with col2:
        stop_button = st.button("Stop Detection")
    with col3:
        clear_button = st.button("Clear Count")

# Static people count display
people_count_placeholder = st.markdown("<h4 style='text-align: center; color: #2980B9;'>People detected: 0</h4>", unsafe_allow_html=True)

# Line chart placeholder
chart_placeholder = st.empty()  # Create an empty placeholder for the chart

# Handle button actions
if start_button:
    st.session_state.running = True
    while st.session_state.running:
        people_count()  # Call the people_count function
        people_count_placeholder.markdown(f"<h4 style='text-align: center; color: #2980B9;'>People detected: {current_people_count}</h4>", unsafe_allow_html=True)

        # Create the Altair chart
        chart = alt.Chart(st.session_state.chart_data).mark_line(point=True).encode(
            x=alt.X('Index', axis=alt.Axis(labels=False, title='')),  # Hide x-axis labels and title
            y=alt.Y('People Count',
                     axis=alt.Axis(tickMinStep=1, format='d'),  # Ensure whole numbers on y-axis
                     scale=alt.Scale(domain=(0, max(st.session_state.chart_data['People Count'], default=0) + 1), nice=True))  # Adjust y-axis scale
        ).properties(
            width=700,
            height=400
        )

        # Display the chart in the placeholder
        chart_placeholder.altair_chart(chart)  # Update the chart with new data
        time.sleep(1)  # Delay for the next detection

if stop_button:
    st.session_state.running = False  # Stop detection

# Clear count functionality
if clear_button:
    st.session_state.latest_count = [0]  # Reset the list with a starting value to avoid empty chart
    st.session_state.chart_data = pd.DataFrame({
        'Index': [0],
        'People Count': [0]  # Keep initial value to avoid empty chart
    })
    people_count_placeholder.markdown("<h3 style='text-align: center; color: #2980B9;'>People detected: 0</h3>", unsafe_allow_html=True)
    st.success("Count cleared!")

# Display the chart when detection is stopped
if not st.session_state.running:
    if not st.session_state.chart_data.empty:
        chart_placeholder.altair_chart(alt.Chart(st.session_state.chart_data).mark_line(point=True).encode(
            x=alt.X('Index', axis=alt.Axis(labels=False, title='')),  # Hide x-axis labels and title
            y=alt.Y('People Count',
                     axis=alt.Axis(tickMinStep=1, format='d'),  # Ensure whole numbers on y-axis
                     scale=alt.Scale(domain=(0, max(st.session_state.chart_data['People Count'], default=0) + 1), nice=True))
        ).properties(
            width=700,
            height=400
        ))
