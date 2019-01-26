import cv2
import numpy as np

img = cv2.imread('/Users/xingoo/Desktop/WechatIMG129.jpeg')
#img = cv2.imread('/Users/xingoo/Desktop/WechatIMG132.jpeg')
#img = cv2.imread('/Users/xingoo/PycharmProjects/ml-in-action/07-实践-其他/31-银行卡识别/1311547459863_.pic_hd.jpg')
h, w, _ = img.shape
img = cv2.resize(img,dsize=(int(w*0.5), int(h*0.5)))
cv2.imshow('原图', img)

# 转灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('灰度图', gray)

blur = cv2.blur(gray, (5, 5))
cv2.imshow('高斯', blur)

canny_threshold = 50
edged = cv2.Canny(blur, canny_threshold, canny_threshold*3)
cv2.imshow('canny', edged)

(_, thresh) = cv2.threshold(edged, 150, 255, cv2.THRESH_BINARY)
cv2.imshow("binary", thresh)

#closed = cv2.erode(thresh, None, iterations = 4)
closed = cv2.dilate(thresh, None, iterations = 6)
cv2.imshow("腐蚀膨胀", closed)

_, contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

contours = sorted(contours, key=cv2.contourArea, reverse=True)

angles = []
for cnt in contours:
    # rect[0]: x,y 表示中心点的坐标
    # rect[1]: w,h 表示高度和宽度
    # rect[2]: angle 表示角度
    rect = cv2.minAreaRect(cnt)
    box = np.int0(cv2.boxPoints(rect))
    cv2.drawContours(img, [box], -1, (0, 255, 0), 3)

    epsilon = 0.051 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    target = approx
    cv2.drawContours(img, [target], -1, (0, 255, 255), 2)

    # if len(approx) == 4:
    #     target = approx
    #     cv2.drawContours(img, [target], -1, (0, 255, 255), 2)

    break

cv2.imshow('result', img)
cv2.waitKey(0)