import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('1.jpeg', 0)

blur = cv2.bilateralFilter(img, 9, 15, 15)

plt.subplot(1, 2, 1)
plt.imshow(img, 'gray')
plt.subplot(1, 2, 2)
plt.imshow(blur, 'gray')

print('end')