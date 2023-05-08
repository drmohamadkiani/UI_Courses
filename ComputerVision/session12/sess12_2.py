import cv2
import numpy as np


cap=cv2.VideoCapture("class.mp4")
facedetector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
vwriter=cv2.VideoWriter('out.wmv',cv2.VideoWriter_fourcc(*'WMV1'),20,(640,480))

while(True):
    ret,frame=cap.read()
    if ret:
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=facedetector.detectMultiScale(gray,1.1,4)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        vwriter.write(frame)
        cv2.imshow('frame',frame)
        cv2.waitKey(10)
    else:
        vwriter.release()


cap.release()