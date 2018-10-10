import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('1.jpeg', 0)
#kernel = np.array([0, -1, 0, -1, 5, -1, 0, -1, 0]).reshape((3, 3))
#kernel = np.array([0, 1, 0, 1, -5, 1, 0, 1, 0]).reshape((3, 3))
#kernel = np.array([1, 1, 1, 1, -8, 1, 1, 1, 1]).reshape((3, 3))
kernel = np.array([-1, -1, -1, -1, 9, -1, -1, -1, -1]).reshape((3, 3))
filter_img = cv2.filter2D(img, -1, kernel)

plt.subplot(1, 2, 1)
plt.imshow(img, 'gray')
plt.subplot(1, 2, 2)
plt.imshow(filter_img, 'gray')
plt.show()
print('end')