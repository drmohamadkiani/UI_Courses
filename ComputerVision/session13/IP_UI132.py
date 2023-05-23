import cv2
import numpy as np
img=cv2.imread('cir.jpg')
cv2.imshow('img',img)
cv2.waitKey(0)
img=cv2.medianBlur(img,5)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


circles=cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,60,param1=40,param2=40,minRadius=10,maxRadius=60)
print(circles.shape)
for i in circles[0,:]:
    cv2.circle(img,(int(i[0]),int(i[1])),int(i[2]),(0,255,0),2)
    cv2.circle(img,(int(i[0]),int(i[1])),3,(0,0,255),3)


cv2.imshow('res',img)
cv2.waitKey(0)