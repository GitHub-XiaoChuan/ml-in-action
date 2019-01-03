import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('1.jpeg', 0)

params = cv2.SimpleBlobDetector_Params()

# params.minThreshold = 10
# params.maxThreshold = 200
#
# params.filterByArea = True
# params.minArea = 1500
#
# params.filterByCircularity = True
# params.minCircularity = 0.1
#
# params.filterByConvexity = True
# params.minConvexity = 0.87
#
# params.filterByInertia = True
# params.minInertiaRatio = 0.01

detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(img)

im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


plt.subplot(1, 2, 1)
plt.imshow(img, 'gray')
plt.subplot(1, 2, 2)
plt.imshow(im_with_keypoints, 'gray')
plt.show()