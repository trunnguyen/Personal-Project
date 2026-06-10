def late_fusion(face_out, audio_out):
    # face_out["label"] and audio_out["label"] are ints

    if face_out["label"] == audio_out["label"]:
        return face_out["label"]

    # fallback: choose higher confidence
    if face_out["confidence"] >= audio_out["confidence"]:
        return face_out["label"]
    else:
        return audio_out["label"]
