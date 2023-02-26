import cv2
import numpy as np

A=cv2.imread('lena.jpg')

# M=np.float32([[1,0,-130],[0,1,120]])#translate
# M=np.float32([[-1,0,512],[0,-1,512]])#mirror
# M=np.float32([[np.cos(np.pi/6),-np.sin(np.pi/6),512],[np.sin(np.pi/6),np.cos(np.pi/6),0]])
# M=np.float32([[1,0.2,0],[1,1,0]])

# pt1=np.float32([[50,50],[200,50],[90,200]])
# pt2=np.float32([[40,90],[139,200],[70,510]])

pt1=np.float32([[50,50],[200,50],[90,200],[120,150]])
pt2=np.float32([[50,90],[200,90],[90,290],[120,220]])


# M=cv2.getAffineTransform(pt1,pt2)
M=cv2.getPerspectiveTransform(pt1,pt2)


print(M)
# r=cv2.warpAffine(A,M,(A.shape[1]*2,A.shape[0]*2))
r=cv2.warpPerspective(A,M,(A.shape[1]*2,A.shape[0]*2))

cv2.imshow('res',r)
cv2.waitKey(0)