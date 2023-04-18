import cv2
import numpy as np
k=3
A=cv2.imread('im.png')
F=np.ones((k,k))/k**2
print(F)


cv2.imshow('image',A)
cv2.waitKey(0)


B=cv2.filter2D(A,-1,F)
cv2.imshow('image2',B)
cv2.waitKey(0)


