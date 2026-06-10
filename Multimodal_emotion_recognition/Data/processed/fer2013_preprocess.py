import os
import cv2
import numpy as np

RAW_DIR = r"C:\Users\nguye\OneDrive\Documents\Data_Science\Multimodal_emotion_recognition\Data\raw\fer2013\train"
OUT_X = "C:/Users/nguye/OneDrive/Documents/Data_Science/Multimodal_emotion_recognition/Data/processed/fer2013/fer2013_faces.npy"
OUT_Y = "C:/Users/nguye/OneDrive/Documents/Data_Science/Multimodal_emotion_recognition/Data/processed/fer2013/fer2013_labels.npy"

EMOTION_MAP = {
    "angry": 0,
    "happy": 1,
    "sad": 2,
    "neutral": 3
}

faces = []
labels = []

for emotion, label in EMOTION_MAP.items():
    emotion_dir = os.path.join(RAW_DIR, emotion)

    if not os.path.exists(emotion_dir):
        print(f"⚠️ Missing folder: {emotion_dir}")
        continue

    for file in os.listdir(emotion_dir):
        if not file.lower().endswith((".jpg", ".png")):
            continue

        path = os.path.join(emotion_dir, file)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.resize(img, (48, 48))
        img = img / 255.0

        faces.append(img)
        labels.append(label)

faces = np.array(faces, dtype=np.float32)
labels = np.array(labels, dtype=np.int64)

np.save(OUT_X, faces)
np.save(OUT_Y, labels)

print(f"✅ FER2013 processed: {faces.shape[0]} samples")