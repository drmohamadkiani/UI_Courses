import cv2
import numpy as np
F=cv2.imread('lena.jpg',0)
cv2.imshow('lena',F)
cv2.waitKey(0)

# k=11
# G=np.ones((k,k))/k**2
# G=np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
# G2=G.T

G=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])


H=cv2.filter2D(F,-1,G)
cv2.imshow('H',H)
cv2.waitKey(0)
#
# H2=cv2.filter2D(F,-1,G2)
# cv2.imshow('H2',H2)
# cv2.waitKey(0)
#
# cv2.imshow('res',H+H2)
# cv2.waitKey(0)
