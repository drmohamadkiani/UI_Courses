import cv2
import numpy as np
k=7
A=cv2.imread('im.png')
# A=cv2.imread('lena.jpg')



cv2.imshow('image',A)
cv2.waitKey(0)

# B=cv2.medianBlur(A,k)
B=cv2.GaussianBlur(A,(k,k),3)
cv2.imshow('image2',B)
cv2.waitKey(0)


