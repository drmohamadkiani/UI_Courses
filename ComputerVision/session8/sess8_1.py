import cv2
import numpy as np


A=cv2.imread('d.jpg',0)
cv2.imshow('image',A)
cv2.waitKey(0)


B=cv2.Canny(A,150,220)

cv2.imshow('res',B)
cv2.waitKey(0)
