import numpy as np
import cv2
from matplotlib import pyplot as plt

"""
ORB是一种想要代替SIFT和SURF的算法，它更快。

1 FAST features from accelerated segment test，绘制16像素的圆
2 BRIEF Binary Robust Independt Elementary Features
3 暴力匹配 
"""

img1 = cv2.imread('4.jpeg', 0)
img2 = cv2.imread('5.jpeg', 0)

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x:x.distance)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:100], img2, flags=2)

plt.imshow(img3)
plt.show()