import cv2
import numpy as np

"""
高斯噪声不同，每个像素点都出现噪声。
在opencv中需要将图像矩阵转换成浮点数再进行加法操作，注意这里用了嵌套的where用于截断小于0和大于255的值
"""

img = cv2.imread('/Users/xingoo/PycharmProjects/ml-in-action/07-实践-其他/36-身份证识别预处理/1.jpeg')
h, w, c = img.shape
scale = 1000/max(h, w)
img = cv2.resize(img, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
cv2.imshow('img', img)

h, w, c = img.shape

img.astype("float")
Gauss_noise = np.random.normal(0, 20, (h, w, c))
Gauss = img + Gauss_noise
Gauss = np.where(Gauss < 0, 0, np.where(Gauss > 255, 255, Gauss))
cv2.imshow("peppers_Gauss", Gauss.astype("uint8"))
cv2.waitKey()

img.astype("float")
Gauss_noise = np.random.normal(0, 50, (h, w, c))
Gauss = img + Gauss_noise
Gauss = np.where(Gauss < 0, 0, np.where(Gauss > 255, 255, Gauss))
cv2.imshow("peppers_Gauss", Gauss.astype("uint8"))
cv2.waitKey()


img.astype("float")
Gauss_noise = np.random.normal(0, 100, (h, w, c))
Gauss = img + Gauss_noise
Gauss = np.where(Gauss < 0, 0, np.where(Gauss > 255, 255, Gauss))
cv2.imshow("peppers_Gauss", Gauss.astype("uint8"))
cv2.waitKey()