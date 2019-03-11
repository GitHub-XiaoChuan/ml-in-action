import cv2
import numpy as np
"""
椒盐噪声出现在随机的像素点位置，而高斯噪声不同，每个像素点都出现噪声
"""
img = cv2.imread('/Users/xingoo/PycharmProjects/ml-in-action/07-实践-其他/36-身份证识别预处理/1.jpeg')
h, w, c = img.shape
scale = 1000/max(h, w)
img = cv2.resize(img, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
cv2.imshow('img', img)

h, w, c = img.shape
noise_salt = np.random.randint(0, 256, (h, w, c))
noise_pepper = np.random.randint(0, 256, (h, w, c))

rand = 0.8
noise_salt = np.where(noise_salt < rand * 256, 255, 0)
noise_pepper = np.where(noise_pepper < rand * 256, -255, 0)

img.astype("float")
noise_salt.astype("float")
noise_pepper.astype("float")
salt = img + noise_salt
pepper = img + noise_pepper
total = img + noise_pepper + noise_salt

salt = np.where(salt > 255, 255, salt)
pepper = np.where(pepper < 0, 0, pepper)
cv2.imshow("salt", salt.astype("uint8"))
cv2.imshow("pepper", pepper.astype("uint8"))
cv2.imshow("total", total.astype("uint8"))
cv2.waitKey()

