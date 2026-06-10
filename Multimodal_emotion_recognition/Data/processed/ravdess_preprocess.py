import os
import librosa
import numpy as np

RAW_DIR = r"C:\Users\nguye\OneDrive\Documents\Data_Science\Multimodal_emotion_recognition\Data\raw\Radvess"
OUT_DIR = r"C:\Users\nguye\OneDrive\Documents\Data_Science\Multimodal_emotion_recognition\Data\processed\ravdess"
EMOTION_MAP = {
    "angry": 0,
    "happy": 1,
    "sad": 2,
    "neutral": 3
}

RAVDESS_LABEL_MAP = {
    "05": "angry",
    "03": "happy",
    "04": "sad",
    "01": "neutral"
}

X, y = [], []

for actor in os.listdir(RAW_DIR):
    actor_dir = os.path.join(RAW_DIR, actor)
    if not os.path.isdir(actor_dir):
        continue

    for file in os.listdir(actor_dir):
        parts = file.split("-")
        if len(parts) < 3:
            continue

        emo_id = parts[2]
        if emo_id not in RAVDESS_LABEL_MAP:
            continue

        path = os.path.join(actor_dir, file)
        audio, sr = librosa.load(path, sr=16000)

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc = np.mean(mfcc.T, axis=0)

        X.append(mfcc)
        y.append(EMOTION_MAP[RAVDESS_LABEL_MAP[emo_id]])

X = np.array(X)
y = np.array(y)

np.save(os.path.join(OUT_DIR, "ravdess_audio.npy"), X)
np.save(os.path.join(OUT_DIR, "ravdess_labels.npy"), y)

print("RAVDESS processed:", len(X))