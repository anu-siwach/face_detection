import streamlit as st
import numpy as np
import cv2, torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from fer import FER
import dlib

# Load models
detector = dlib.get_frontal_face_detector()
emotion_detector = FER()
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

st.title("🔍 AI Face Analyzer")
uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    faces = detector(gray)

    if faces:
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            crop = img_np[y:y+h, x:x+w]
            face_pil = Image.fromarray(crop)

            inputs = processor(images=face_pil, return_tensors="pt")
            with torch.no_grad():
                caption = model.generate(**inputs)
            description = processor.decode(caption[0], skip_special_tokens=True)

            emotion_result = emotion_detector.detect_emotions(crop)
            emotion = max(emotion_result[0]["emotions"], key=emotion_result[0]["emotions"].get) if emotion_result else "Neutral"

            st.image(image, caption="Uploaded", use_column_width=True)
            st.subheader("Description:")
            st.write(description)
            st.subheader("Emotion:")
            st.write(emotion)
    else:
        st.error("No face detected.")
