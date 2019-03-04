import cv2
import numpy as np

path = '1551431027.9676428.jpg'
path = '1551431762.311429.jpg'
path = '1551431803.881511.jpg'
path = '1551431818.712335.jpg'
image = cv2.imread(path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)
#cv2.waitKey(0)

gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)
cv2.imshow("g", gradient)
#cv2.waitKey(0)

blurred = cv2.blur(gradient, (9, 9)) # 9*9的核做模糊
(_, thresh) = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
cv2.imshow("binary", thresh)
#cv2.waitKey(0)

closed = cv2.erode(thresh, None, iterations=2)
closed = cv2.dilate(closed, None, iterations=20)
cv2.imshow("f", closed)
#cv2.waitKey(0)

(_, cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c = sorted(cnts, key=cv2.contourArea, reverse=True)

for cnt in c:
    # rect[0]: x,y 表示中心点的坐标
    # rect[1]: w,h 表示高度和宽度
    # rect[2]: angle 表示角度
    rect = cv2.minAreaRect(cnt)
    box = np.int0(cv2.boxPoints(rect))

    angle = rect[2]
    x,y = rect[0]
    w,h = rect[1]
    image_h,image_w,_ = image.shape
    cv2.drawContours(image, [box], -1, (0, 255, 0), 3)

    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = box

    # 顺序分别为 (x2, y2), (x3, y3), (x4, y4), `(x1, y1)
    new_image = image[min(y2, y3): max(y1, y4), min(x2, x1):max(x3, x4), :]
    cv2.imshow('q', new_image)

    break

cv2.imshow('r', image)
cv2.waitKey(0)