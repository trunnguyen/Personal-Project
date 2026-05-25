import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture(0)
P_Time=0
mp_Face=mp.solutions.face_detection
mp_Draw=mp.solutions.drawing_utils
Face_Detection=mp_Face.FaceDetection(0.75)
while True:
    success,img=cap.read()
    Img_RGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result = Face_Detection.process(Img_RGB)
    print(result)

    if result.detections:
        for id,detection in enumerate(result.detections):
            mp_Draw.draw_detection(img,detection)
            # print(id,detection)
            # print(detection.score)
            print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            ih,iw,ic=img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img,bbox,(255,0,255),2)
            cv2.putText(img, f':{int(detection.score[0]*100)}%',
                        (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 2)

    C_Time=time.time()
    fps=1/(C_Time-P_Time)
    P_Time=C_Time
    cv2.putText(img,f'FPS:{int(fps)}',(20,70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,(255,0,0),2)
    cv2.imshow("Image",img)
    cv2.waitKey(1)
