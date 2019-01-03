import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('1.jpeg', 0)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# 形态梯度
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

# 礼帽
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

# 黑帽
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

plt.subplot(2,2,1)
plt.imshow(img,'gray')
plt.title('src')
plt.subplot(2,2,2)
plt.imshow(gradient,'gray')
plt.title('sobelX')
plt.subplot(2,2,3)
plt.imshow(tophat,'gray')
plt.title('sobelY')
plt.subplot(2,2,4)
plt.imshow(blackhat,'gray')
plt.title('sobelXY')
plt.show()