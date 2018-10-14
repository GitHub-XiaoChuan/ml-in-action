import cv2
import numpy as np

# hough变换
img = cv2.imread('4.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 120)
minLineLength = 10
maxLineGap = 5
# 处理的图像；线段的几何表示；阈值
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 40, minLineLength, maxLineGap)

for line in lines:
    line = line[0]
    cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)

cv2.imshow('edges', edges)
cv2.imshow('lines', img)

cv2.waitKey(0)