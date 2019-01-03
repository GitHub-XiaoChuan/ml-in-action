import cv2
import numpy
import matplotlib.pyplot as plt

img = cv2.imread('1.jpeg', 0)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilate = cv2.dilate(img, kernel)
erode = cv2.erode(img, kernel)

result = cv2.absdiff(dilate, erode)
#cv2.imshow('1', result)

retval, result = cv2.threshold(result, 40, 255, cv2.THRESH_BINARY)
#cv2.imshow('2', result)

result = cv2.bitwise_not(result)

plt.subplot(2,2,1)
plt.imshow(img,'gray')
plt.title('src')
plt.subplot(2,2,2)
plt.imshow(dilate,'gray')
plt.title('sobelX')
plt.subplot(2,2,3)
plt.imshow(erode,'gray')
plt.title('sobelY')
plt.subplot(2,2,4)
plt.imshow(result,'gray')
plt.title('sobelXY')
plt.show()