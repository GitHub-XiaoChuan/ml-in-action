import cv2
import numpy as np
import matplotlib.pyplot as plt

min_width = 20
min_height = min_width

image = cv2.imread("010.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("灰度图", gray)

# compute the Scharr gradient magnitude representation of the images
# in both the x and y direction
# 使用Scharr操作（指定使用ksize = -1）构造灰度图在水平和竖直方向上的梯度幅值表示
gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

# subtract the y-gradient from the x-gradient
# Scharr操作之后，我们从x-gradient中减去y-gradient，通过这一步减法操作，最终得到包含高水平梯度和低竖直梯度的图像区域。
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)

cv2.imshow("梯度", gradient)

blurred = cv2.blur(gradient, (9, 9)) # 9*9的核做模糊
(_, thresh) = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
cv2.imshow("binary", thresh)

# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
# closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
# cv2.imshow("morphologyex", closed)

closed = cv2.erode(thresh, None, iterations = 4)
closed = cv2.dilate(closed, None, iterations = 6)
cv2.imshow("腐蚀膨胀", closed)


(_, cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c = sorted(cnts, key=cv2.contourArea, reverse=True)
angles = []
for cnt in c:
    # rect[0]: x,y 表示中心点的坐标
    # rect[1]: w,h 表示高度和宽度
    # rect[2]: angle 表示角度

    rect = cv2.minAreaRect(cnt)
    box = np.int0(cv2.boxPoints(rect))

    angle = rect[2]
    w,h = rect[1]
    if angle not in map(float,[0,90,-90]) and w>min_width and h>min_height:
        print("%d %d %d" % (w, h, angle))
        if w < h:
            if angle > 0:
                angle = angle - 90
            else:
                angle = angle + 90

        # print(angle)
        # print(w>h)
        angles.append(angle)
        # if w > h:
        #     cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
        # else:
        #     cv2.drawContours(image, [box], -1, (0, 0, 255), 3)

cv2.imshow("before", image)

angles = np.array(sorted(angles))
print(angles)

positive_number = len(angles[np.where(angles>0)])
if positive_number/len(angles) > 0.5:
    angles = angles[np.where(angles>0)]
else:
    angles = angles[np.where(angles<0)]



# 进行中间比例取样
start = np.percentile(angles, 30)
end = np.percentile(angles, 60)
print(start)
print(end)


angles = np.array(angles[np.where(angles>start)])
angles = np.array(angles[np.where(angles<end)])


mean_angle = np.mean(angles)
print(mean_angle)

rows, cols, _ = image.shape
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), mean_angle, 1)
image = cv2.warpAffine(image, M, (cols, rows))


#rotate = image.transpose(mean_angle)

cv2.imshow("after", image)
cv2.imwrite("after.jpg", image)
cv2.waitKey(0)

