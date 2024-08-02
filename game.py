import mediapipe as mp
import cv2
import numpy as np
import time
import random
from mediapipe.framework.formats import landmark_pb2

media_drawing=mp.solutions.drawing_utils
hands=mp.solutions.hands
score=0

enemy_1=random.randint(50,600)
enemy_2=random.randint(50,400)

def enemy():
    global score,enemy_1,enemy_2
    cv2.circle(image,(enemy_1,enemy_2),25,(1,255,0),-1)
    
capture=cv2.VideoCapture(0)
cv2.namedWindow('Hand Tracking',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hand Tracking',1200,720)
with hands.Hands(min_detection_confidence=0.8,min_tracking_confidence=0.5)as hand: #21 sensitive points in hand while detecting and also that will be called in further as hand
    while capture.isOpened():
        _,frame=capture.read()
        
        image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)#opencv works as bgr where as mediapipe works on rgb
        image=cv2.flip(image,1)
        
        imageHeight,imageWidth,_=image.shape
        output=hand.process(image)
        image=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        
        
        font=cv2.FONT_HERSHEY_SIMPLEX
        color=(255,0,255)
        text=cv2.putText(image,"Score",(480,30),font,1,color,4,cv2.LINE_AA)
        text=cv2.putText(image,str(score),(590,30),font,1,color,4,cv2.LINE_AA)
        enemy()
        
        if output.multi_hand_landmarks:
            for num,hand_landmarks in enumerate(output.multi_hand_landmarks):
                media_drawing.draw_landmarks(image,hand_landmarks,mp.solutions.hands.HAND_CONNECTIONS,media_drawing.DrawingSpec(color=(250,14,250),thickness=2,circle_radius=2),media_drawing.DrawingSpec(color=(250,14,250),thickness=2,circle_radius=2))
        
        
        if output.multi_hand_landmarks :
            for handLandmarks in output.multi_hand_landmarks :
                for point in hands.HandLandmark:
                    normalizedLandmark=handLandmarks.landmark[point]
                    pixelCoordinatesLandmark=media_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x,normalizedLandmark.y,imageWidth,imageHeight)
                    point=str(point) #they are originally dictionary to make them use we convert them into string
                    if point=='HandLandmark.INDEX_FINGER_TIP':
                        try:
                            cv2.circle(image,(pixelCoordinatesLandmark[0],pixelCoordinatesLandmark[1]),25,(10,255,0),5)
                            if pixelCoordinatesLandmark[0]==enemy_1 or pixelCoordinatesLandmark[0]==enemy_1+10 or pixelCoordinatesLandmark[0]==enemy_1 -10:
                                print("found")
                                enemy_1=random.randint(50,600)
                                enemy_2=random.randint(50,400)
                                score=score+1
                                font=cv2.FONT_HERSHEY_SIMPLEX
                                color=(255,0,255)
                                text=cv2.putText(image,"Score",(480,30),font,1,color,4,cv2.LINE_AA)
                                enemy()
                        except:
                            pass
        cv2.imshow('Hand Tracking',image)
        if cv2.waitKey(10) & 0xFF==ord("k"):
            print(score)
            break
capture.release()
cv2.destroyAllWindows()
            
                                                                                
        
        
    