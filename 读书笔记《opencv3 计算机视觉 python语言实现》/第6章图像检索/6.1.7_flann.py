import numpy as np
import cv2
import matplotlib.pyplot as plt

queryImage = cv2.imread('4.jpeg')
trainingImage = cv2.imread('5.jpeg')

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(queryImage, None)
kp2, des2 = orb.detectAndCompute(trainingImage, None)

# FLANN_INDEX_KDTREE = 1
#
# indexParams = dict(algrithm=FLANN_INDEX_KDTREE, trees=5)
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,  # 12
                    key_size=12,  # 20
                    multi_probe_level=1)  # 2
searchParams = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, searchParams)

matches = flann.knnMatch(des1, des2, k=2)

matchesMask = [[0, 0] for i in range(len(matches))]

for i, x in enumerate(matches):
    if len(x) == 2 and x[0].distance < 0.7 * x[1].distance:
            matchesMask[i] = [1, 0]

drawParams = dict(matchColor=(0, 255, 0),
                  singlePointColor=(255, 0, 0),
                  matchesMask=matchesMask,
                  flags=0)

resultImage = cv2.drawMatchesKnn(queryImage, kp1, trainingImage, kp2, matches, None, **drawParams)
plt.imshow(resultImage)
plt.show()
