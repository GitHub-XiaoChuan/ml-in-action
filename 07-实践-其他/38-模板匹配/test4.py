import cv2
import numpy as np
path = '/Users/xingoo/PycharmProjects/ml-in-action/07-实践-其他/38-模板匹配/mask/idcard_mask.jpg'

mask_img = cv2.imread(path)
h, w, _ = mask_img.shape
scale = 640 / w
mask_img = cv2.resize(mask_img, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

target = '/Users/xingoo/Documents/dataset/身份证图片/sfz_201703_03_F12C3EE9-67BF-42D5-950C-13B1A573D024.jpg'
target_img = cv2.imread(target)
h, w, _ = target_img.shape
scale = 1920 / w
target_img = cv2.resize(target_img, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(mask_img, None)
kp2, des2 = sift.detectAndCompute(target_img, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=10)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# store all the good matches as per Lowe's ratio test.
# 两个最佳匹配之间距离需要大于ratio 0.7,距离过于相似可能是噪声点
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)
# reshape为(x,y)数组
if len(good) > 10:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    # 用HomoGraphy计算图像与图像之间映射关系, M为转换矩阵
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    # 使用转换矩阵M计算出img1在img2的对应形状
    h, w, _ = mask_img.shape
    M_r = np.linalg.inv(M)
    im_r = cv2.warpPerspective(target_img, M_r, (w, h))
