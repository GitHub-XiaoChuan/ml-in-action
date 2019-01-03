import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('1.jpeg', 0)# 直接读取灰度图像
img1 = np.float32(img)

# 均值滤波
kernel = np.ones((5, 5), np.float32)/9
print(kernel)

dst = cv2.filter2D(img1, -1, kernel)
plt.subplot(1, 2, 1)
plt.imshow(img1, 'gray')
plt.subplot(1, 2, 2)
plt.imshow(dst, 'gray')

print('end')