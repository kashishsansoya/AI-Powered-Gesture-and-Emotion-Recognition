import streamlit as st
import cv2
import numpy as np
from PIL import Image
from gtts import gTTS
import base64
import os
from utils import HuggingFacePredictor

# Page configuration
st.set_page_config(
    page_title="Emotion & ISL Assistant - HuggingFace",
    page_icon="👐",
    layout="wide"
)

# Initialize predictor
@st.cache_resource
def load_predictor():
    return HuggingFacePredictor()

predictor = load_predictor()

def text_to_speech(text):
    """Convert text to speech"""
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("output.mp3")
        return "output.mp3"
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return None

def autoplay_audio(audio_file):
    """Auto-play audio"""
    try:
        with open(audio_file, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f'<audio controls autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
            st.markdown(md, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Audio Error: {e}")

# Main App
st.markdown('<div class="main-header">🎭 Emotion & ISL Assistant - HuggingFace</div>', unsafe_allow_html=True)



# Mode selection
mode = st.radio("Choose input mode:", ["📷 Webcam Live", "📁 Upload Image"], horizontal=True)

if mode == "📷 Webcam Live":
    st.markdown("### 🎥 Live Webcam Feed")
    
    # Webcam access
    img_file_buffer = st.camera_input("Take a picture with your webcam for real-time analysis")
    
    if img_file_buffer is not None:
        # Read image from buffer
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Process frame
        with st.spinner("🤖 AI is analyzing in real-time..."):
            emotions, gestures, annotated_frame = predictor.process_frame(cv2_img)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 🎨 Annotated Live View")
            st.markdown('<div class="webcam-container">', unsafe_allow_html=True)
            annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, use_column_width=True, 
                    caption="Live detection with MediaPipe face and hand landmarks")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 🔍 Real-Time Results")
            
            # Display emotions
            if emotions:
                for emotion, confidence, bbox in emotions:
                    st.markdown(f'<div class="emotion-card">', unsafe_allow_html=True)
                    st.write(f"**Emotion:** {emotion}")
                    st.write(f"**Confidence:** {confidence:.1%}")
                    st.markdown(f'<div class="confidence-bar" style="width: {min(confidence*100, 100)}%"></div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("❌ No face detected - try facing the camera directly")
            
            # Display gestures
            if gestures:
                for gesture, confidence in gestures:
                    if gesture != "No Hand":
                        st.markdown(f'<div class="gesture-card">', unsafe_allow_html=True)
                        st.write(f"**Gesture:** {gesture}")
                        st.write(f"**Confidence:** {confidence:.1%}")
                        st.markdown(f'<div class="confidence-bar" style="width: {min(confidence*100, 100)}%"></div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
            
            # Text-to-Speech
            if emotions or gestures:
                emotion_text = emotions[0][0] if emotions else "No emotion detected"
                gesture_text = gestures[0][0] if gestures and gestures[0][0] != "No Hand" else "No gesture detected"
                
                prediction_text = f"Real-time analysis complete. {emotion_text}. {gesture_text}."
                
                if st.button("🔊 Speak Results", type="primary"):
                    with st.spinner("Generating audio..."):
                        audio_file = text_to_speech(prediction_text)
                        if audio_file:
                            st.success("✅ Audio generated! Click play:")
                            autoplay_audio(audio_file)

else:  # Upload Image mode
    uploaded_file = st.file_uploader("📁 Upload an image with face and hands", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("🔍 AI Analysis")
            
            # Convert image for processing
            image_np = np.array(image)
            if len(image_np.shape) == 3:
                if image_np.shape[2] == 3:
                    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                elif image_np.shape[2] == 4:
                    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
            
            # Process image
            with st.spinner("🤖 AI is analyzing..."):
                emotions, gestures, annotated_image = predictor.process_frame(image_bgr)
            
            # Display emotions
            if emotions:
                for emotion, confidence, bbox in emotions:
                    st.markdown(f'<div class="emotion-card">', unsafe_allow_html=True)
                    st.write(f"**Emotion:** {emotion}")
                    st.write(f"**Confidence:** {confidence:.1%}")
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("❌ No face detected in the image")
            
            # Display gestures
            if gestures:
                for gesture, confidence in gestures:
                    if gesture != "No Hand":
                        st.markdown(f'<div class="gesture-card">', unsafe_allow_html=True)
                        st.write(f"**Gesture:** {gesture}")
                        st.write(f"**Confidence:** {confidence:.1%}")
                        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show annotated image
        if annotated_image is not None:
            st.subheader("🎨 Detection Results")
            annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, use_column_width=True, 
                    caption="AI detection with face bounding boxes and hand landmarks")
