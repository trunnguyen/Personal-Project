    import streamlit as st
    import cv2
    import numpy as np
    import torch
    import tempfile
    import sys
    import os

    ROOT_DIR = r'C:\Users\nguye\OneDrive\Documents\Data_Science\Multimodal_emotion_recognition'

    if ROOT_DIR not in sys.path:
        sys.path.append(ROOT_DIR)

    from inference.pipeline import multimodal_predict
    from models.face_emotion.cnn import FaceCNN
    from models.speech_emotion.lstm import SpeechLSTM
    from inference.audio_infer import extract_audio_features
    from inference.face_infer import preprocess_face

    EMOTION_MAP = {
        0: "Angry",
        1: "Happy",
        2: "Sad",
        3: "Neutral"
    }

    @st.cache_resource
    def load_models():
        face_model = FaceCNN(num_classes=4)
        audio_model = SpeechLSTM(num_classes=4)

        face_model.eval()
        audio_model.eval()

        return face_model, audio_model



    face_model, audio_model = load_models()

    st.set_page_config(page_title="Multimodal Emotion Recognition", layout="centered")
    st.title("🎭 Multimodal Emotion Recognition")
    st.write("Upload an image and an audio file to analyze emotions.")

    img_file = st.file_uploader("📷 Upload a face image", type=["jpg", "png"])
    audio_file = st.file_uploader("🎙️ Upload an audio file", type=["wav"])


    if img_file and audio_file:

        img_bytes = np.frombuffer(img_file.getvalue(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        st.image(img, caption="Uploaded Image", channels="BGR")

        face_tensor = preprocess_face(img)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            audio_path = tmp.name

        audio_feat = extract_audio_features(audio_path)

        result = multimodal_predict(
            face_tensor=face_tensor,
            audio_features=audio_feat,
            face_model=face_model,
            audio_model=audio_model
        )

        st.subheader("🎯 Final Emotion")

        """st.success(
            f"{result['final']['label']} "
            f"(confidence: {result['final']['confidence']:.2f})"
        )"""

        final_label = result["final"]
        emotion_label = EMOTION_MAP.get(final_label, "Unknown")

        st.success(f"🎯 Predicted Emotion: {emotion_label}")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🙂 Face Emotion")
            face_label = EMOTION_MAP.get(result["face"]["label"], "Unknown")
            st.write(face_label)
            st.progress(float(result["face"]["confidence"]))

        with col2:
            st.subheader("🎙️ Audio Emotion")
            audio_label = EMOTION_MAP.get(result["audio"]["label"], "Unknown")
            st.write(audio_label)
            st.progress(float(result["audio"]["confidence"]))


    else:
        st.info("Please upload both an image and an audio file to start.")
