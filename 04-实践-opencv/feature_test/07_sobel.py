import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('1.jpeg', 0)

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobelXY = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)

plt.subplot(2,2,1)
plt.imshow(img,'gray')
plt.title('src')
plt.subplot(2,2,2)
plt.imshow(sobelx,'gray')
plt.title('sobelX')
plt.subplot(2,2,3)
plt.imshow(sobely,'gray')
plt.title('sobelY')
plt.subplot(2,2,4)
plt.imshow(sobelXY,'gray')
plt.title('sobelXY')
plt.show()