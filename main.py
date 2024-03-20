import cv2
import base64
import av
import matplotlib.pyplot as plt
import pandas as pd
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

csv_file_path = "counts_vid_seconds.csv"
df = pd.read_csv(csv_file_path)


st.markdown(
    """
<style>
    .cover-glow {
        width: 100%;
        height: auto;
        padding: 3px;
        box-shadow: 
            0 0 5px #330000,
            0 0 10px #660000,
            0 0 15px #990000,
            0 0 20px #CC0000,
            0 0 25px #FF0000,
            0 0 30px #FF3333,
            0 0 35px #FF6666;
        position: relative;
        z-index: -1;
        border-radius: 30px;  /* Rounded corners */
    }
</style>
    """,
    unsafe_allow_html=True,
)

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Function to convert image to base64
def img_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
 



# Load and display sidebar image with glowing effect
img_path = "smartislogo.png"
img_base64 = img_to_base64(img_path)
st.sidebar.markdown(
    f'<img src="data:image/png;base64,{img_base64}" class="cover-glow">',
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")



# Sidebar for Mode Selection
mode = st.sidebar.radio("Select Mode:", options=["Live detection", "Single angle", "Multiple angles", "Data insight"], index=0)
st.sidebar.markdown("---")


 
if mode == "Live detection":
    st.title("Webcam Live Feed")
    def process(image):
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                            int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return cv2.flip(image, 1)

    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    class VideoProcessor:
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img = process(img)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False}
    )

    
 
elif mode == "Single angle":
    st.title("Object Detection and Tracking - Single angle")
    vid1 = "output_video.mp4"
    st.video(vid1)
    
 
elif mode == "Multiple angles":
    st.title("Object Detection and Tracking - Multiple angles")
    vid2 = "hall1hall2.mp4"
    st.video(vid2)
 
elif mode == "Data insight":
    st.title("Object Detection and Tracking - Data Insight")
    vid3 = "output_video.mp4"
    st.video(vid3)
 
    fig, ax1 = plt.subplots()
    ax1.plot(df["Time (seconds).1"], df["Player Count"], color="blue")
    ax1.set_xlabel("Time (seconds).1")
    ax1.set_ylabel("Player Count")
    ax1.set_title("Player Count vs Time")
    ax1.set_yticks(range(0,15))
 
    st.pyplot(fig)
 
    # Plot 2: Line plot of Bystander Count vs Time
    fig, ax2 = plt.subplots()
    ax2.plot(df["Time (seconds).1"], df["Bystander Count"], color="green")
    ax2.set_xlabel("Time (seconds).1")
    ax2.set_ylabel("Bystander Count")
    ax2.set_title("Bystander Count vs Time")
    ax2.set_yticks(range(0, 15))
 
    st.pyplot(fig)
 
    # Plot 3: Bar graph of Minimum and Maximum Player Count and Bystander Count
    min_max_data = {
        "Category": ["Minimum", "Maximum"],
        "Player Count": [df["Player Count"].min(), df["Player Count"].max()],
        "Bystander Count": [df["Bystander Count"].min(), df["Bystander Count"].max()]
    }
 
    min_max_df = pd.DataFrame(min_max_data)
 
    fig, ax3 = plt.subplots()
    min_max_df.plot(kind="bar", x="Category", ax=ax3)
    ax3.set_ylabel("Count")
    ax3.set_title("Minimum and Maximum Player Count and Bystander Count")
 
    st.pyplot(fig)




