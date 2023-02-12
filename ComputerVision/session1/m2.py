import cv2
import numpy as np

A=cv2.imread('a.jpg')
h,w,c=A.shape
cv2.imshow('image',A)
cv2.waitKey(0)

B=cv2.imread('8.jpg')
B=cv2.resize(B,(w,h))
cv2.imshow('image2',B)
cv2.waitKey(0)



# C=np.vstack((A,B))
A=A.astype(float)
B=B.astype(float)

C=A*B
C=C/np.max(C)
C*=255
C=C.astype(np.uint8)

cv2.imshow('result',C)
cv2.waitKey(0)




