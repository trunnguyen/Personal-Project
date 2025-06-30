import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture(0)
cap.set(3,1980)
cap.set(4,1080)
pTime=0
mp_draw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
FaceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec=mp_draw.DrawingSpec(thickness=1, circle_radius=1)
while(True):
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = FaceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for FaceLm in results.multi_face_landmarks:
            mp_draw.draw_landmarks(img,FaceLm,
                                   mpFaceMesh.FACEMESH_CONTOURS,
                                   drawSpec,drawSpec)
            for id,lm in enumerate(FaceLm.landmark):
                # print(lm)
                ih, iw, ic =img.shape
                x,y =int(lm.x*iw),int(lm.y*ih)
                print(id,x,y)






    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,f'FPS: {int(fps)}',(20,70),cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 3)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
