import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, staticMode=False, maxHands=2, minDetectionCon=0.5, trackCon=0.5):
        self.staticMode = staticMode
        self.minDetectionCon = max(0.0, min(1.0, minDetectionCon))
        self.minTrackCon = max(0.0, min(1.0, trackCon))
        self.maxHands = maxHands
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.staticMode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.minDetectionCon,
            min_tracking_confidence=self.minTrackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)


        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:

                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmlist = []

        if self.results.multi_hand_landmarks:

            if handNo < len(self.results.multi_hand_landmarks):
                myhand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myhand.landmark):

                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    lmlist.append([id, cx, cy])
                    if draw:

                        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
            else:

                pass

        return lmlist


def main():
    pTime = 0
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    detector = handDetector()

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame. Exiting...")
            break

        img = detector.findHands(img)
        lmlist = detector.findPosition(img)

        if len(lmlist) != 0:

            if len(lmlist) > 4:
                print(lmlist[4])
            else:
                print("Less than 5 landmarks detected for the first hand.")

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()