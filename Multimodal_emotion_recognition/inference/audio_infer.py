import librosa
import numpy as np
import torch
import torch.nn.functional as F


def extract_audio_features(
    audio_path,
    sample_rate=16000,
    n_mfcc=40,
    max_len=300
):
    """
    Extract MFCC features from an audio file.
    Returns: Tensor (1, T, F)
    """

    y, sr = librosa.load(audio_path, sr=sample_rate)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = mfcc.T  # (T, F)

    if mfcc.shape[0] < max_len:
        pad = max_len - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad), (0, 0)))
    else:
        mfcc = mfcc[:max_len]

    mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)
    return mfcc_tensor  # (1, T, F)


def predict_audio(audio_features, model):
    """
    Run speech emotion model inference.
    Returns dict with label + confidence.
    """

    # ✅ FORCE correct shape (B, T, F)
    if audio_features.dim() == 2:
        audio_features = audio_features.unsqueeze(0)
    elif audio_features.dim() == 4:
        audio_features = audio_features.squeeze(1)

    model.eval()
    with torch.no_grad():
        logits = model(audio_features)  # (B, num_classes)
        probs = F.softmax(logits, dim=1)

        label = torch.argmax(probs, dim=1).item()
        confidence = probs[0, label].item()

    return {
        "label": label,
        "confidence": confidence
    }
