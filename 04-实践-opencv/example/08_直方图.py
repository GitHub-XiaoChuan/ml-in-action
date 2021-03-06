import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/Users/xingoo/Desktop/img/1551681101.7306502.jpg', 0) #直接读为灰度图像
#opencv方法读取-cv2.calcHist（速度最快）
#图像，通道[0]-灰度图，掩膜-无，灰度级，像素范围
hist_cv = cv2.calcHist([img],[0],None,[256],[0,256])
#numpy方法读取-np.histogram()
hist_np,bins = np.histogram(img.ravel(),256,[0,256])
#numpy的另一种方法读取-np.bincount()（速度=10倍法2）
hist_np2 = np.bincount(img.ravel(),minlength=256)
plt.subplot(221),plt.imshow(img,'gray')
plt.subplot(222),plt.plot(hist_cv)
plt.subplot(223),plt.plot(hist_np)
plt.subplot(224),plt.plot(hist_np2)
plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('/Users/xingoo/Desktop/img/1551681101.7306502.jpg',0) #直接读为灰度图像
mask = np.zeros(img.shape[:2],np.uint8)
mask[100:200,100:200] = 255
masked_img = cv2.bitwise_and(img,img,mask=mask)

#opencv方法读取-cv2.calcHist（速度最快）
#图像，通道[0]-灰度图，掩膜-无，灰度级，像素范围
hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])

plt.subplot(221),plt.imshow(img,'gray')
plt.subplot(222),plt.imshow(mask,'gray')
plt.subplot(223),plt.imshow(masked_img,'gray')
plt.subplot(224),plt.plot(hist_full),plt.plot(hist_mask)
plt.show()

img = cv2.imread('/Users/xingoo/Desktop/img/1551681101.7306502.jpg',0) #直接读为灰度图像
res = cv2.equalizeHist(img)

plt.subplot(121),plt.imshow(img,'gray')
plt.subplot(122),plt.imshow(res,'gray')
plt.show()

import cv2
import matplotlib.pyplot as plt

img = cv2.imread('/Users/xingoo/Desktop/img/1551681101.7306502.jpg',0) #直接读为灰度图像
clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(10,10))
cl1 = clahe.apply(img)

plt.subplot(121),plt.imshow(img,'gray')
plt.subplot(122),plt.imshow(cl1,'gray')
plt.show()