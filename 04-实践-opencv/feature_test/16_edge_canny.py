import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('1.jpeg', 0)

# 超过150认为是边缘、小于30认为不是边缘
canny = cv2.Canny(img, 30, 150)

plt.subplot(1, 2, 1)
plt.imshow(img, 'gray')
plt.subplot(1, 2, 2)
plt.imshow(canny, 'gray')
plt.show()