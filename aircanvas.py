#Handson air canvas
import cv2
import numpy as np 
import mediapipe as mp
from collections import deque

#Initialize Red colour
rpoints=deque(maxlen=1024)

paintWindow=np.zeros((471,636,3))+255
paintWindow=cv2.rectangle(paintWindow,(40,1),(140,65),(0,0,0),2)

cv2.putText(paintWindow,"CLEAR",(49,33),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)

#Initialize mediapipe library
mpHands=mp.solutions.hands
hands=mpHands.Hands(max_num_hands=1,min_detection_confidence=0.7)
mpDraw=mp.solutions.drawing_utils

#webcam
cap=cv2.VideoCapture(0)
while True:
    ignore, frame=cap.read()
    x,y,c=frame.shape

    frame=cv2.flip(frame,1)
    framergb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    frame=cv2.rectangle(frame,(40,1),(140,65),(0,0,0),2)
    cv2.putText(frame,"CLEAR",(49,33),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)

    result=hands.process(framergb)

    if result.multi_hand_landmarks:
        landmarks=[]
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx=int(lm.x*640)
                lmy=int(lm.y*480)
                landmarks.append([lmx,lmy])
            mpDraw.draw_landmarks(frame,handslms,mpHands.HAND_CONNECTIONS)
        centre=(landmarks[8][0],landmarks[8][1])
        if centre[1] <=65 and 40<=centre[0] <=140:
            rpoints.clear()
            paintWindow[67:, :]=255
        elif centre[1]>65:
            rpoints.appendleft(centre)
    
    for k in range(1,len(rpoints)):
        if rpoints[k-1] is None or rpoints[k] is None:
            continue
        cv2.line(frame,rpoints[k-1],rpoints[k],(0,0,255),20)
        cv2.line(paintWindow,rpoints[k-1],rpoints[k],(0,0,255),2)

    cv2.imshow("Output",frame)
    cv2.imshow("Paint",paintWindow)
    if cv2.waitKey(1)==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()