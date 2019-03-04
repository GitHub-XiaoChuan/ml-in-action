import cv2
import matplotlib.pyplot as plt

img = cv2.imread('/Users/xingoo/Desktop/img/1551681101.7306502.jpg',0) #直接读为灰度图像
clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(10,10))
cl1 = clahe.apply(img)

plt.subplot(121),plt.imshow(img,'gray')
plt.subplot(122),plt.imshow(cl1,'gray')
plt.show()

import cv2
import matplotlib.pyplot as plt

img = cv2.imread('/Users/xingoo/Desktop/img/1551681101.7306502.jpg',0) #直接读为灰度图像
clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(20,20))
cl1 = clahe.apply(img)

plt.subplot(121),plt.imshow(img,'gray')
plt.subplot(122),plt.imshow(cl1,'gray')
plt.show()

import cv2
import matplotlib.pyplot as plt

img = cv2.imread('/Users/xingoo/Desktop/img/1551681101.7306502.jpg',0) #直接读为灰度图像
clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(30,30))
cl1 = clahe.apply(img)

plt.subplot(121),plt.imshow(img,'gray')
plt.subplot(122),plt.imshow(cl1,'gray')
plt.show()