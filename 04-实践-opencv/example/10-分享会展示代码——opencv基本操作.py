import cv2

path = '/Users/xingoo/Desktop/img/1551681020.594172.jpg'
path = '/Users/xingoo/Desktop/img/1551680929.439567.jpg'

# 加载图像
img = cv2.imread(path)

# 获得长和宽
h, w, _ = img.shape
print(img.shape)
cv2.imshow('img', img)
cv2.waitKey(0)

# 调整大小
scale = 700/max(h, w)
resize_img = cv2.resize(img, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
print(resize_img.shape)
cv2.imshow('resize_img', resize_img)
cv2.waitKey(0)

# 灰度图
gray = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)
cv2.waitKey(0)

# 边缘检测
gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)
cv2.imshow("gradient", gradient)
cv2.waitKey(0)

# 高斯模糊 9*9的核做模糊
blurred = cv2.blur(gradient, (9, 9))
cv2.imshow('blur', blurred)
cv2.waitKey(0)

(_, thresh) = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
cv2.imshow("binary", thresh)
cv2.waitKey(0)

# 服饰膨胀
closed = cv2.erode(thresh, None, iterations=3)
cv2.imshow("erode", closed)
closed = cv2.dilate(closed, None, iterations=3)
cv2.imshow("dilate", closed)
cv2.waitKey(0)
