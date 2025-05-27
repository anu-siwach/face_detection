import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from fer import FER

# Face detector
face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt",
                                    "res10_300x300_ssd_iter_140000.caffemodel")

# Caption model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model     = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Emotion detector
fer_detector = FER()

st.title("Face Analyzer")

uploaded = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])
if not uploaded:
    st.info("Upload a file to proceed.")
    st.stop()

image = Image.open(uploaded).convert("RGB")
img_np = np.array(image)
st.image(image, use_container_width=True)

# Detect faces
h, w = img_np.shape[:2]
blob = cv2.dnn.blobFromImage(img_np, 1.0, (300,300), (104,177,123))
face_net.setInput(blob)
dets = face_net.forward()

# Take highest-confidence face
best = max(
    ((d[2], (d[3:7]*[w,h,w,h]).astype(int)) for d in dets[0,0]),
    key=lambda x: x[0]
)
conf, box = best
if conf < 0.5:
    st.error("No face with high-enough confidence detected.")
    st.stop()

x1,y1,x2,y2 = box
# expand margin
dx,dy = int(0.1*(x2-x1)), int(0.1*(y2-y1))
x1,y1 = max(0,x1-dx), max(0,y1-dy)
x2,y2 = min(w,x2+dx), min(h,y2+dy)

crop = img_np[y1:y2, x1:x2]
face_pil = Image.fromarray(crop)

# Caption
inputs = processor(images=face_pil, return_tensors="pt")
cap_ids = model.generate(**inputs,
                         num_beams=5,
                         no_repeat_ngram_size=2,
                         top_p=0.9,
                         temperature=0.8,
                         max_new_tokens=50)
description = processor.decode(cap_ids[0], skip_special_tokens=True).capitalize()

# Emotion (FER only)
fer_res = fer_detector.detect_emotions(crop)
if fer_res:
    emotions = fer_res[0]["emotions"]
    top_emotion = max(emotions, key=emotions.get)
    emotion = top_emotion.capitalize()
else:
    emotion = "Neutral"

# Display
st.subheader("Detected Face")
st.image(face_pil, width=300)
st.markdown(f"**Description:** {description}")
st.markdown(f"**Emotion:** {emotion}")
