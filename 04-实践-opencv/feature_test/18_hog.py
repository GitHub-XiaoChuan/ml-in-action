import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('1.jpeg', 0)

hist = cv2.HOGDescriptor().compute(img, (8, 8), (8, 8))
#hist = hist.reshape((-1,))


plt.subplot(1, 2, 1)
plt.imshow(img, 'gray')
plt.subplot(1, 2, 2)
plt.hist(hist)
plt.show()