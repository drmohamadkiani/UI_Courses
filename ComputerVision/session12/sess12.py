import cv2
import numpy as np


cap=cv2.VideoCapture(0)
# fgbg=cv2.createBackgroundSubtractorMOG2()

while(True):
    ret,frame=cap.read()
    if ret:
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame=cv2.Canny(frame,0,255)
        # frame=fgbg.apply(frame)

        cv2.imshow('frame',frame)
        cv2.waitKey(20)


cap.release()