import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('1.jpeg', 0)

lap = cv2.Laplacian(img, cv2.CV_64F)
lap = np.uint8(np.absolute(lap))

plt.subplot(1, 2, 1)
plt.imshow(img,'gray')
plt.subplot(1, 2, 2)
plt.imshow(lap, 'gray')
plt.show()