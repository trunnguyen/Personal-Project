from inference.face_infer import predict_face
from inference.audio_infer import predict_audio
from models.fusion.late_fusion import late_fusion


def multimodal_predict(face_tensor, audio_features, face_model, audio_model):
    face_result = predict_face(face_model, face_tensor)
    audio_result = predict_audio(audio_features, audio_model)

    final_label = late_fusion(face_result, audio_result)

    return {
        "face": face_result,
        "audio": audio_result,
        "final": final_label
    }

