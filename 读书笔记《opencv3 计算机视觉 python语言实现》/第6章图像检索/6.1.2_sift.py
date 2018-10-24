import cv2
import sys
import numpy as np

"""
scale invariant feature transform SIFT尺度不变特征变换
"""

img = cv2.imread('1.jpeg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
keypoints, descriptor = sift.detectAndCompute(gray, None)

img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                        color=(51, 163, 236))

cv2.imshow('sift', img)
while True:
    if cv2.waitKey(0):
        break
cv2.destroyAllWindows()