import cv2
import mediapipe as mp
import time


class FaceDetector:
    def __init__(self,min_detection_confidence=0.5):
        self.min_detection_confidence = min_detection_confidence
        self.mp_Face = mp.solutions.face_detection
        self.mp_Draw = mp.solutions.drawing_utils
        self.Face_Detection = self.mp_Face.FaceDetection(self.min_detection_confidence)

    def find_face(self,img,draw=True):
        Img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.Face_Detection.process(Img_RGB)
        print(self.result)
        bboxs=[]

        if self.result.detections:
            for id,detection in enumerate(self.result.detections):
                self.mp_Draw.draw_detection(img,detection)
                # print(id,detection)
                # print(detection.score)
                print(detection.location_data.relative_bounding_box)
                bboxC = detection.location_data.relative_bounding_box
                ih,iw,ic=img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id,bbox,detection.score])
                if draw:
                    img=self.Draw(img,bbox)
                    cv2.putText(img, f':{int(detection.score[0]*100)}%',(bbox[0],bbox[1]-20),
                                cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2)
        return img,bboxs
    def Draw(self,img,bbox, l=30, t=10):
        x, y, w, h = bbox
        x1,y1 = x+w,y+h
        cv2.rectangle(img, bbox, (255, 0, 255), 2)

        cv2.line(img,(x,y),(x+l,y),(255,0,255),t)
        cv2.line(img, (x, y), (x , y+l), (255, 0, 255), t)

        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y - l), (255, 0, 255), t)

        cv2.line(img, (x, y1), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y - l), (255, 0, 255), t)

        cv2.line(img, (x, y1), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x1, y - l), (255, 0, 255), t)


def main():
    cap = cv2.VideoCapture(0)
    P_Time = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img,bboxs=detector.find_face(img)
        C_Time = time.time()
        fps = 1 / (C_Time - P_Time)
        P_Time = C_Time
        cv2.putText(img, f'FPS:{int(fps)}', (20, 70),cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__=='__main__':
    main()