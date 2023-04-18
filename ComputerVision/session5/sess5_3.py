import cv2
import numpy as np


A=cv2.imread('lena.jpg',0)
cv2.imshow('image',A)
cv2.waitKey(0)

# F=np.zeros((3,3))
# F[0,:]=1
# F[2,:]=-1

# F2=F.T
#
# B=cv2.filter2D(A,0,F)
# cv2.imshow('res',B)
# cv2.waitKey(0)
#
# C=cv2.filter2D(A,0,F2)
# cv2.imshow('res2',C)
# cv2.waitKey(0)

L=np.array([[0,1,0],[1,-4,1],[0,1,0]])
LI=cv2.filter2D(A,0,L)
cv2.imshow('res',LI)
cv2.waitKey(0)

