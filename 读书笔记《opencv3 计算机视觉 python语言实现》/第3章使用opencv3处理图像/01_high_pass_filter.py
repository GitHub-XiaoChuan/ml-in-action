import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

kernel_3x3 = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])
print(kernel_3x3)

kernel_5x5 = np.array([[-1, -1, -1, -1, -1],
                       [-1,  1,  2,  1, -1],
                       [-1,  2,  4,  2, -1],
                       [-1,  1,  2,  1, -1],
                       [-1, -1, -1, -1, -1]])
print(kernel_5x5)

img = cv2.imread("../2.jpeg", 0)

print(np.shape(img))
print(kernel_5x5.shape)

# 3x3的核
k3 = ndimage.convolve(img, kernel_3x3)

# 5x5的核
k5 = ndimage.convolve(img, kernel_5x5)

# 高斯滤波，低通滤波做差，也能得到高通滤波
blurred = cv2.GaussianBlur(img, (11, 11), 0)
g_hpf = img - blurred

plt.subplot(2, 2, 1)
plt.imshow(img, 'gray')
plt.subplot(2, 2, 2)
plt.imshow(k3, 'gray')
plt.subplot(2, 2, 3)
plt.imshow(k5, 'gray')
plt.subplot(2, 2, 4)
plt.imshow(g_hpf, 'gray')
plt.show()
