import cv2
import numpy as np
from matplotlib import pyplot as plt


img=cv2.imread('lena.jpg',0)

f=np.fft.fft2(img)
f=np.fft.fftshift(f)
absf=np.abs(f)
logabs=200*np.log(absf)
plt.subplot(321)
plt.imshow(img,cmap='gray')
plt.subplot(322)
plt.imshow(logabs,cmap='gray')
#
# #lowpass
bw=100
rows,cols=img.shape
crow,ccol=rows//2,cols//2
ze=np.zeros((rows,cols))
ze[crow-bw:crow+bw,ccol-bw:ccol+bw]=1
f1=f*ze
#
absf=np.abs(f1)
logabs=200*np.log(absf)
plt.subplot(323)
plt.imshow(logabs,cmap='gray')
#
imrecover=np.fft.ifftshift(f1)
imrecover=np.fft.ifft2(imrecover)
image=np.abs(imrecover)
plt.subplot(324)
plt.imshow(image,cmap='gray')
image=image.astype(np.uint8)
# cv2.imshow('res',image)
# cv2.waitKey(0)
#
#
#
# #highpass
bw=10
rows,cols=img.shape
crow,ccol=rows//2,cols//2
ze=np.ones((rows,cols))
ze[crow-bw:crow+bw,ccol-bw:ccol+bw]=0
f2=f*ze

absf=np.abs(f2)
logabs=200*np.log(absf)
plt.subplot(326)
plt.imshow(logabs,cmap='gray')

imrecover=np.fft.ifftshift(f2)
imrecover=np.fft.ifft2(imrecover)
image=np.abs(imrecover)
plt.subplot(325)
plt.imshow(image,cmap='gray')
image=image.astype(np.uint8)
cv2.imshow('res2',image)
cv2.waitKey(0)
#

plt.show()