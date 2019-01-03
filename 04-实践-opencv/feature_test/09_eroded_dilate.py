import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('1.jpeg', 0)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
print(kernel)

# eroded = cv2.erode(img, kernel)
# dilated = cv2.dilate(img, kernel)
eroded = cv2.erode(img, None)
dilated = cv2.dilate(img, None)
eroded_dilated = cv2.dilate(dilated, kernel)

plt.subplot(2,2,1)
plt.imshow(img,'gray')
plt.title('src')
plt.subplot(2,2,2)
plt.imshow(eroded,'gray')
plt.title('sobelX')
plt.subplot(2,2,3)
plt.imshow(dilated,'gray')
plt.title('sobelY')
plt.subplot(2,2,4)
plt.imshow(eroded_dilated,'gray')
plt.title('sobelXY')
plt.show()