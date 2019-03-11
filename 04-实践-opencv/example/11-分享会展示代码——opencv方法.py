import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(100, 100))
# cl1 = clahe.apply(gray)
# cv2.imshow('clahe', cl1)

# 边缘检测
gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)
cv2.imshow("gradient", gradient)
cv2.waitKey(0)

# 高斯模糊 9*9的核做模糊
blurred = cv2.blur(gradient, (9, 9))
(_, thresh) = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
cv2.imshow("binary", thresh)
cv2.waitKey(0)

# 服饰膨胀
closed = cv2.erode(thresh, None, iterations=2)
cv2.imshow("erode", closed)
closed = cv2.dilate(closed, None, iterations=2)
cv2.imshow("dilate", closed)
cv2.waitKey(0)

thresh = closed

# 获得水平和竖直方向的像素均值
h, w = thresh.shape
hor_mean = [np.array(thresh[i, :]).mean() for i in range(h)]


# 输出函数曲线图，这里输出了水平方向和竖直方向上的像素情况，可以很明显的知道数字的分布

hor_mean_reverse = hor_mean.copy()
hor_mean_reverse.reverse()

rows, cols = thresh.shape
X = np.arange(0, rows, 1)

plt.subplot(121)
plt.imshow(thresh, 'gray')
plt.subplot(122)
plt.plot(hor_mean_reverse, X)
plt.show()


# 计算切割线
thresh_value = 10
thresh_w = 20
current_w = 0
current_begin = 0

buffer = []

for current, mean in enumerate(hor_mean):
    if mean > thresh_value:
        current_w += 1
    else:
        if current_w > thresh_w:
            #print('%d - %d' % (current_begin, current))
            buffer.append((current_begin, current))

        current_begin = current
        current_w = 0

h, w, _ = resize_img.shape
for begin, end in buffer:
    begin -= 10
    end += 10
    cv2.line(resize_img, (0, begin), (w - 1, begin), (0, 255, 0), 1)  # 5
    cv2.line(resize_img, (0, end), (w - 1, end), (0, 255, 0), 1)  # 5

cv2.imshow('r', resize_img)
cv2.waitKey(0)

# 获得身份证行
card_begin, card_end = buffer[-1]
new_img = img[card_begin-10:card_end+10, :, :]
cv2.imshow('card', new_img)

new_thresh = thresh[card_begin-10:card_end+10, :]
cv2.imshow('new_thresh', new_thresh)
cv2.waitKey(0)

# 获得竖直方向投影
card_h, card_w = new_thresh.shape
ver_mean = [np.array(new_thresh[:, i]).mean() for i in range(card_w)]

plt.subplot(211)
plt.imshow(new_thresh, 'gray')
plt.subplot(212)
X = range(0, card_w, 1)
plt.plot(X, ver_mean)
plt.show()

thresh_value = 10
thresh_w = 8
current_w = 0
current_begin = 0

buffer = []
for current, mean in enumerate(ver_mean):
    if mean > thresh_value:
        current_w += 1
    else:
        if current_w > thresh_w:
            print('%d - %d' % (current_begin, current))
            buffer.append((current_begin, current))

        current_begin = current
        current_w = 0

h, w, _ = resize_img.shape
for begin, end in buffer:
    # begin -= 5
    # end += 5
    cv2.line(resize_img, (begin, 0), (begin, h-1), (255, 255, 0), 1)  # 5
    cv2.line(resize_img, (end, 0), (end, h-1), (255, 255, 0), 1)  # 5

    cv2.rectangle(resize_img, (begin-5, card_begin), (end+5, card_end), (255, 0, 255), 3)

cv2.imshow('r', resize_img)
cv2.waitKey(0)