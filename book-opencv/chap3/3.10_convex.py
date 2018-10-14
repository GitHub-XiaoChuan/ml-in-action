import cv2
import numpy as np

# 多边形逼近

# 1.找到轮廓
img = cv2.imread('2.jpg', 0)

_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
image, contours, hier = cv2.findContours(thresh, 3, 2)
cnt = contours[0]

# 2.找到角点
epsilon = 0.01 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)

# 3. 绘制图形
image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.polylines(image, [approx], True, (0, 255, 0), 2)
print(len(approx))

cv2.imshow('approx', image)
cv2.waitKey(0)


# 凸包
# 1.寻找轮廓
img = cv2.imread('2.jpg', 0)

_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
image, contours, hier = cv2.findContours(thresh, 3, 2)
cnt = contours[0]

# 2. 寻找凸包角点
hull = cv2.convexHull(cnt)

# 3. 绘制
image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.polylines(image, [hull], True, (0, 255, 0), 2)
cv2.imshow('convex hull', image)
cv2.waitKey(0)

cv2.imshow('img', image)
cv2.waitKey(0)