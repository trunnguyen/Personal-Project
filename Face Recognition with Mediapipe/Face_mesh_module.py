import cv2
import mediapipe as mp
import time


class FaceMeshDetector:  # Renamed for clarity to avoid confusion with mp.solutions.face_mesh.FaceMesh
    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = bool(staticMode)
        self.minDetectionCon = max(0.0, min(1.0, minDetectionCon))
        self.minTrackCon = max(0.0, min(1.0, minTrackCon))
        self.maxFaces = maxFaces

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh_model = self.mp_face_mesh.FaceMesh(
            static_image_mode=self.staticMode,
            max_num_faces=self.maxFaces,
            min_detection_confidence=self.minDetectionCon,
            min_tracking_confidence=self.minTrackCon
        )
        self.drawSpec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))  # Green color

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = self.face_mesh_model.process(self.imgRGB)

        faces_data = []
        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(
                        image=img,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=self.drawSpec,
                        connection_drawing_spec=self.drawSpec
                    )

                current_face_landmarks = []
                for id, lm in enumerate(face_landmarks.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    # cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                    # print(id, x, y) # This will print a lot of data; uncomment for debugging specific points
                    current_face_landmarks.append([x, y])
                faces_data.append(current_face_landmarks)

        return img, faces_data


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    pTime = 0
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame.")
            break

        img, faces = detector.findFaceMesh(img, True)

        if len(faces) != 0:
            print(f"First face landmarks count: {len(faces[0])}, Example: {faces[0][0]}")

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()