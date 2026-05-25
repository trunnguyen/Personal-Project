import cv2
from deepface import DeepFace

faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open camera")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    dominant_emotion = "Detecting..."

    if len(faces) > 0:
        try:
            result = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="opencv"  # 🔥 much faster
            )
            dominant_emotion = result[0]["dominant_emotion"]
        except Exception as e:
            dominant_emotion = "Error"

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(
        frame,
        dominant_emotion,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 0, 255),
        2
    )

    cv2.imshow("Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
