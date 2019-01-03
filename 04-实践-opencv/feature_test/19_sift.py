import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('1.jpeg', 0)
img2 = cv2.resize(img, (300, 300))


sift = cv2.SIFT()

plt.subplot(1, 2, 1)
plt.imshow(img, 'gray')
plt.subplot(1, 2, 2)
plt.imshow(img2, 'gray')
plt.show()