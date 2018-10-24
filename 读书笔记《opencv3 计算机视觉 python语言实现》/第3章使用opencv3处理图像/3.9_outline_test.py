import cv2
import numpy as np

# pyrDown 先对图像进行高斯平滑，然后再进行降采样（将图像尺寸行和列方向缩减一半）
img = cv2.pyrDown(cv2.imread('1.jpg', cv2.IMREAD_UNCHANGED))

cv2.imshow('img', img)
cv2.waitKey(0)

ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
image, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    # 找到最小的平行矩形
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # 计算最小的矩形
    rect = cv2.minAreaRect(c)
    # calculate coordinates of the minimum area rectangle
    box = cv2.boxPoints(rect)
    box_d = np.int0(box)
    # 第三个参数，如果是-1，表示绘制所有的形状；如果是其他的值，代表绘制对应的图形
    cv2.drawContours(img, [box_d], 0, (0, 0, 255), 3)

    # 计算最小的圆
    (x, y), radius = cv2.minEnclosingCircle(c)
    center = (int(x), int(y))
    radius = int(radius)

    cv2.circle(img, center, radius, (0, 255, 0), 2)

# 画出所有的点
cv2.drawContours(img, contours, -1, (255, 0, 0), 1)

cv2.imshow('img', img)
cv2.waitKey(0)