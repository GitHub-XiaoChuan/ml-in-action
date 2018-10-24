import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('4.jpeg', 0)
img2 = cv2.imread('5.jpeg', 0)

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.knnMatch(des1, des2, 1)
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, img2, flags=2)

plt.imshow(img3)
plt.show()