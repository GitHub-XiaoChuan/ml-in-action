import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../2.jpeg', 0)

blur = cv2.medianBlur(img, 7)

cv2.Laplacian(blur, cv2.CV_8U, blur, 5)

# 不是很理解这里的归一化是在干嘛
normalizedInverseAlpha = (1.0 / 255) * (255 - blur)
channels = cv2.split(img)
for channel in channels:
    channel[:] = channel * normalizedInverseAlpha

plt.subplot(2, 2, 1)
plt.imshow(img, 'gray')
plt.subplot(2, 2, 2)
plt.imshow(blur, 'gray')
plt.show()