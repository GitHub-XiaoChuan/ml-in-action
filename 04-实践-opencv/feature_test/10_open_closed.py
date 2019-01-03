import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('1.jpeg', 0)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
all = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

plt.subplot(2,2,1)
plt.imshow(img,'gray')
plt.title('src')
plt.subplot(2,2,2)
plt.imshow(closed,'gray')
plt.title('sobelX')
plt.subplot(2,2,3)
plt.imshow(opened,'gray')
plt.title('sobelY')
plt.subplot(2,2,4)
plt.imshow(all,'gray')
plt.title('sobelXY')
plt.show()