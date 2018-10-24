import numpy as np
import cv2
import matplotlib.pyplot as plt

MIN_MATCH_COUNT = 10

img1 = cv2.imread('4.jpeg', 0)
img1 = cv2.resize(img1, (0, 0), fx=0.5, fy=0.5)
img2 = cv2.imread('5.jpeg', 0)

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

FLANN_INDEX_LSH = 6

index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,  # 12
                    key_size=12,  # 20
                    multi_probe_level=1)  # 2
searchParams = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, searchParams)

matches = flann.knnMatch(des1, des2, k=2)

good = []

for i, x in enumerate(matches):
    if len(x) == 2 and x[0].distance < 0.7 * x[1].distance:
            good.append(x[0])

if len(good) >= MIN_MATCH_COUNT:
    # 寻找关键点
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # 单应性
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
else:
    print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
    matchesMask = None

drawParams = dict(matchColor=(0, 255, 0),
                  singlePointColor=(255, 0, 0),
                  matchesMask=matchesMask,
                  flags=2)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **drawParams)
plt.imshow(img3, 'gray')
plt.show()