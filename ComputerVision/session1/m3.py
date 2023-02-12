import cv2
import numpy as np

A=cv2.imread('a.jpg')
h,w,c=A.shape
cv2.imshow('image',A)
cv2.waitKey(0)
# B=A.copy()
# A[:,:w//2]=B[:,w//2:]
# A[:,w//2:]=B[:,:w//2]
A=255-A


# B=cv2.imread('8.jpg')
# B=cv2.resize(B,(w,h))
# cv2.imshow('image2',B)
# cv2.waitKey(0)
#
# A[100:400,200:300]=B[:300,100:200]
cv2.imshow('image',A)
cv2.waitKey(0)