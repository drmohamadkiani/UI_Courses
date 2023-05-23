import cv2
import numpy as np
img=cv2.imread('8.jpg')
cv2.imshow('img',img)
cv2.waitKey(0)
img=cv2.medianBlur(img,5)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges=cv2.Canny(gray,0,255)

cv2.imshow('edges',edges)
cv2.waitKey(0)

lines=cv2.HoughLines(edges,1,np.pi/180,120)
print(lines.shape)
for L in lines:
    rho=L[0][0]
    theta=L[0][1]
    if (theta<np.pi/180*10):
    # if (np.pi/180*80<theta<np.pi/180*110):

        a=np.cos(theta)
        b=np.sin(theta)
        x0=a*rho
        y0=b*rho

        x1=int(x0+1000*(-b))
        y1=int(y0+1000*(a))

        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imshow('res',img)
cv2.waitKey(0)