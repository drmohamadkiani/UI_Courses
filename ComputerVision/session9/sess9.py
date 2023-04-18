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

#lowpass
bw=200
rows,cols=img.shape
crow,ccol=rows//2,cols//2
ze=np.zeros((rows,cols))
ze[crow-bw:crow+bw,ccol-bw:ccol+bw]=1
f1=f*ze

absf=np.abs(f1)
logabs=200*np.log(absf)
plt.subplot(324)
plt.imshow(logabs,cmap='gray')

imrecover=np.fft.ifftshift(f1)
imrecover=np.fft.ifft2(f1)
image=np.abs(imrecover)
plt.subplot(323)
plt.imshow(image,cmap='gray')



#highpass
bw=50
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
imrecover=np.fft.ifft2(f2)
image=np.abs(imrecover)
plt.subplot(325)
plt.imshow(image,cmap='gray')


plt.show()