import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Function to process pose estimation
def process_frame(frame):
    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Detect pose
    results = pose.process(rgb_frame)
    # Draw pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3, circle_radius=2),
        )
    return frame

# Streamlit UI
st.title("Human Pose Estimation using Machine Learning")
st.sidebar.title("Options")
mode = st.sidebar.selectbox("Choose Mode", ["About", "Upload Image", "Use Webcam", "Upload Video"])

# Mode 1: About
if mode == "About":
    st.write("""
    #### What is Pose Estimation?
    Pose estimation is a computer vision technique for tracking the movements of a person or an object. 
    It is usually performed by finding the location of key points for the given objects. 
    We can compare various movements and postures based on these key points and draw insights. 
    Pose estimation is widely used in applications like **augmented reality**, **animation**, **gaming**, and **robotics**.
    """)


 # Display the uploaded image along with the description
    pose_image = Image.open("pose_pic.jpg")  # Load the uploaded image
    st.image(pose_image, caption="Pose Estimation Example", use_container_width=True)

    st.write("""
    #### What You Can Do Here:
    - **Upload Image**: Upload an image file, and the app will detect and display human poses in the image.
    - **Use Webcam**: Activate your webcam to view real-time pose estimation.
    - **Upload Video**: Upload a video file, and the app will process it to detect poses frame-by-frame.

    #### Behind the Scenes:
    - **Pose Detection**: This is powered by MediaPipe, a machine learning framework designed for efficient real-time tracking of human poses.
    - **Visualization**: OpenCV is used to render the detected pose landmarks on the media.

    #### How to Use:
    - Choose a mode from the left sidebar:
      1. **Upload Image**: Simply upload a JPEG/PNG image.
      2. **Use Webcam**: Give permissions to access the webcam and click "Start Webcam."
      3. **Upload Video**: Select a supported video format like MP4 or AVI.
    - View the processed results in the app interface.

    ---
    This application showcases the power of computer vision by analyzing human poses in **images**, **videos**, 
    or **real-time webcam feeds** using **MediaPipe** and **OpenCV**.
    """)

# Mode 2: Upload Image Mode
elif mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file))
        result_image = process_frame(image.copy())
        st.image([image, result_image], caption=["Original", "Pose Estimation"], use_container_width=True)

# Mode 3: Use Webcam
elif mode == "Use Webcam":
    st.text("Using webcam requires permissions and may only work on your local machine.")
    if st.button("Start Webcam"):
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = process_frame(frame)
            stframe.image(frame, channels="BGR")
        cap.release()



# Mode 4: Upload a Video
elif mode == "Upload Video":
    video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if video_file is not None:
        tfile = open("temp_video.mp4", "wb")
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = process_frame(frame)
            stframe.image(frame, channels="BGR")
        cap.release()

st.sidebar.info("Powered by OpenCV and MediaPipe")
