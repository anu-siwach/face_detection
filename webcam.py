import cv2, torch, dlib
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from fer import FER

detector = dlib.get_frontal_face_detector()
emotion_detector = FER()
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        crop = frame[y:y+h, x:x+w]
        face_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        inputs = processor(images=face_pil, return_tensors="pt")
        with torch.no_grad():
            caption = model.generate(**inputs)
        description = processor.decode(caption[0], skip_special_tokens=True)

        emotion_result = emotion_detector.detect_emotions(crop)
        emotion = max(emotion_result[0]["emotions"], key=emotion_result[0]["emotions"].get) if emotion_result else "Neutral"

        cv2.putText(frame, f"{description} ({emotion})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Live Face Analyzer", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
