import cv2
import numpy as np

import Hand_Tracking_module as htm
import time
import os


brushThichNess = 15
eraserThichNess = 50


folderPath=(r"C:\Users\nguye\OneDrive\Documents\Data Analysis\Hand Tracking\Headers")
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image =cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
header=overlayList[0]
drawColor=(255,0,255)

cap=cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector=htm.handDetector(minDetectionCon=0.85)

xp,yp=0,0

imgCanvas=np.zeros((720,1280,3),np.uint8)

while True:

    #1. Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)
    #2. Find Hand Markers
    img= detector.findHands(img)
    lmlist = detector.findPosition(img,draw=False)
    if len(lmlist) != 0:
        fingers = detector.fingersUp()
        print("Fingers:", fingers)
    if len(lmlist) != 0:
        print(lmlist)

        #tips of index and middle
        x1,y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]

        #3. Check which fingers are up
        fingers= detector.fingersUp()
        print(fingers)
        #4. If selection mode - Two finger are up
        if fingers[1] == 1 and fingers[2] ==1:
            xp, yp = 0, 0
            # print("selection Mode")
            if y1< 125:
                if 200<x1<400:
                    header=overlayList[0]
                    drawColor=(255,0,255)
                elif 500< x1 <600:
                    header=overlayList[1]
                    drawColor=(255,0,0)
                elif 700< x1 <800:
                    header=overlayList[2]
                    drawColor=(0,255,0)
                elif 900< x1 <1050:
                    header=overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
        #5. If  drawing mode - index finger are up
        if fingers[1] == 1 and fingers[2]==0:
            cv2.circle(img,(x1,y1),15,drawColor,cv2.FILLED)
            print("Drawing Mode")
            if xp==0 and yp==0:
                xp,yp=x1,y1

            if drawColor == (0,0,0,):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThichNess)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThichNess)

            cv2.line(img,(xp,yp),(x1,y1),drawColor,brushThichNess)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThichNess)

            xp,yp=x1,y1


    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,imgInv =cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY)
    imgInv=cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img=cv2.bitwise_and(img,imgInv)
    img=cv2.bitwise_or(img,imgCanvas)



    #Setting the header image
    img[0:125,0:1280]=header
    img=cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv2.imshow('Image', img)
    # cv2.imshow('Image Canvas', imgCanvas)
    # cv2.imshow('Inv', imgInv)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break