import cv2
import torch
import numpy as np


def preprocess_face(
    img_bgr,
    img_size=48
):
    """
    Preprocess face image for FER2013-style CNN.
    Input: BGR image (OpenCV)
    Output: Tensor (1, 1, 48, 48)
    """

    # Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Resize
    gray = cv2.resize(gray, (img_size, img_size))

    # Normalize
    gray = gray / 255.0

    # Shape: (1, 1, H, W)
    tensor = torch.tensor(gray, dtype=torch.float32)
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    return tensor


import torch
import torch.nn.functional as F

def predict_face(model, face_tensor):
    model.eval()

    with torch.no_grad():
        logits = model(face_tensor)
        probs = F.softmax(logits, dim=1)

        label = torch.argmax(probs, dim=1).item()
        confidence = probs[0, label].item()

    return {
        "label": label,
        "confidence": confidence
    }
