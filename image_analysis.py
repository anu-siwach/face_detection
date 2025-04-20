import os, cv2, torch, pandas as pd, textwrap
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from fer import FER

# Paths
deploy_prototxt = "deploy.prototxt"
model_weights = "res10_300x300_ssd_iter_140000.caffemodel"
folder_path = "Images"
output_csv = "face_descriptions.csv"
no_faces_csv = "no_faces_detected.csv"

# Load models
face_net = cv2.dnn.readNetFromCaffe(deploy_prototxt, model_weights)
emotion_detector = FER()
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

results, no_face_results = [], []

for filename in os.listdir(folder_path):
    if filename.endswith((".jpg", ".png")):
        path = os.path.join(folder_path, filename)
        image = cv2.imread(path)

        if image is None:
            print(f"Error loading {filename}")
            continue

        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104, 177, 123))
        face_net.setInput(blob)
        detections = face_net.forward()

        face_detected = False
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                face_detected = True
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                x, y, x_max, y_max = box.astype("int")
                face_crop = image[y:y_max, x:x_max]

                if face_crop.size == 0: continue

                face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                inputs = processor(images=face_pil, return_tensors="pt")
                with torch.no_grad():
                    caption = model.generate(**inputs)
                description = processor.decode(caption[0], skip_special_tokens=True)

                emotion_result = emotion_detector.detect_emotions(face_crop)
                emotion = max(emotion_result[0]["emotions"], key=emotion_result[0]["emotions"].get) if emotion_result else "Neutral"

                results.append({"Filename": filename, "Description": description, "Emotion": emotion})

        if not face_detected:
            no_face_results.append({"Filename": filename})

cv2.destroyAllWindows()
pd.DataFrame(results).to_csv(output_csv, index=False)
pd.DataFrame(no_face_results).to_csv(no_faces_csv, index=False)
print("Image analysis complete.")
