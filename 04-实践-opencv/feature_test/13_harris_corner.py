import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('1.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# blockSIze是角点检测中的邻域值
# ksize是sobel求偏导的窗口大小
# k是角点检测参数，取值为0.04到0.06
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

dst1 = cv2.dilate(dst, None)

img[dst > 0.00001 * dst.max()] = [255,0,0]

plt.subplot(2,2,1)
plt.imshow(gray,'gray')
plt.title('src')
plt.subplot(2,2,2)
plt.imshow(dst,'gray')
plt.title('sobelX')
plt.subplot(2,2,3)
plt.imshow(dst1,'gray')
plt.title('sobelY')
plt.subplot(2,2,4)
plt.imshow(img,'gray')
plt.title('sobelXY')
plt.show()