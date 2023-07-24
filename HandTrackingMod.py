import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self,node=False,maxHand=2,detectionCon=0.5,trackCon=0.5):
        self.node=node
        self.maxHand=maxHand
        self.detectionCon=detectionCon
        self.trackingCon=trackCon

        self.mpHands=mp.solutions.hands
        self.hands=self.mpHands.Hands()
        self.mpDraw=mp.solutions.drawing_utils

    def findHands(self,img,draw=True):
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self,img,handNo=0,draw=False):

        lmList=[]

        if self.results.multi_hand_landmarks:
            myhand=self.results.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myhand.landmark):
                #print(id,lm)
                h,w,c=img.shape
                cx,cy=int(lm.x*w), int(lm.y*h)
                #print(id,cx,cy)
                lmList.append([id,cx,cy])
                if  draw:
                    cv2.circle(img,(cx,cy),5,(0,0,255),cv2.FILLED)
        return lmList

def main():
    cTime=0
    pTime=0
    cap=cv2.VideoCapture(0)
    detector=handDetector()
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
        





if __name__=="__main__":
    main()