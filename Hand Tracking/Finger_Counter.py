import cv2
import time
import os
import Hand_Tracking_module as htm


wCam, hCam = 640, 480

cap=cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0

folderPath= (r"C:\Users\nguye\OneDrive\Documents\Data Analysis\Hand Tracking\Finger Images")
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    # print(f'{folderPath}/{imPath} ')
    overlayList.append(image)

print(len(overlayList))

detector = htm.handDetector(minDetectionCon=0.5)
tipIds = (4, 8, 12, 16, 20)

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img,draw=False)
    # print(lmList)
    if len(lmList) != 0:
        fingers = []

        #Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        #Fingers
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers=fingers.count(1)
        print(totalFingers)
        if 0 <= totalFingers <= 5:
            overlaySize = 150
            overlayImg = cv2.resize(overlayList[totalFingers - 1], (overlaySize, overlaySize))
            img[0:overlaySize, 0:overlaySize] = overlayImg

        cv2.rectangle(img,(20,225),(170,425),(0,255,0),cv2.FILLED)
        cv2.putText(img,str(totalFingers),(45,375),cv2.FONT_HERSHEY_PLAIN,10,(255,255,255),3)


    cTime = time.time()
    fps=1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f"FPS:{int(fps)} ",(400,70),
                cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2 )

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break