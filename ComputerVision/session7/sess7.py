import cv2
import numpy as np
from matplotlib import pyplot as plt

A=cv2.imread('f.jpg',0)

ax1=plt.subplot(3,2,1)
ax1.imshow(A,cmap='gray')

ax2=plt.subplot(3,2,2)
ax2.hist(A.flatten(),256,[0,255],color='r')



eq=cv2.equalizeHist(A)
ax3=plt.subplot(3,2,3)
ax3.imshow(eq,cmap='gray')

ax4=plt.subplot(3,2,4)
ax4.hist(eq.flatten(),256,[0,255],color='r')


clahe=cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8))
cl=clahe.apply(A)
ax5=plt.subplot(3,2,5)
ax5.imshow(cl,cmap='gray')

ax6=plt.subplot(3,2,6)
ax6.hist(cl.flatten(),256,[0,255],color='r')

plt.show()