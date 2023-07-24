import cv2
import mediapipe as mp
import time
import HandTrackingMod as HTM

cTime=0
pTime=0
cap=cv2.VideoCapture(0)
detector=HTM.handDetector()
while True:
    success,img=cap.read()
    img=detector.findHands(img)
    lmList=detector.findPosition(img)
    if len(lmList) != 0:
        print(lmList[4])
    cTime= time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img,str("HAND TRACKING"),(200,25),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)        
    cv2.putText(img,str(fps),(18,70),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)        
    cv2.putText(img,str("fps:"),(18,25),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)        
    cv2.imshow("Image",img)
    cv2.waitKey(1)