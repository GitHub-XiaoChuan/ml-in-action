import cv2
import numpy as np
import imutils
import numpy as np
from matplotlib import pyplot as plt

"""
基于传统的方式实现身份证ocr

https://yq.aliyun.com/articles/547689
"""

def resize_im(im, scale, max_scale=None):
    """
    调整图片尺寸

    :param im:
    :param scale:
    :param max_scale:
    :return:
    """
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    # 基于opencv调整图片尺寸，使用的方法是inter_linear
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f

def radon_angle(img, angle_split=5):
    angles_list = list(np.arange(-90., 90. + angle_split,
                                 angle_split))

    # 创建一个列表 angles_map_max，存放各个方向上投影的积分最大
    # 值。我们对每个旋转角度进行计算，获得每个角度下图像的投影，
    # 然后计算当前指定角度投影值积分的最大值。最大积分值对应的角度
    # 即为偏转角度。

    angles_map_max = []
    for current_angle in angles_list:
        rotated_img = imutils.rotate_bound(img, current_angle)
        current_map = np.sum(rotated_img, axis=1)
        angles_map_max.append(np.max(current_map))

    adjust_angle = angles_list[np.argmax(angles_map_max)]

    return adjust_angle


path = '/Users/xingoo/Desktop/img/1551681101.7306502.jpg'
path = '/Users/xingoo/Desktop/img/1551681020.594172.jpg'
path = '/Users/xingoo/Desktop/img/1551680929.439567.jpg'
path = '/Users/xingoo/PycharmProjects/ml-in-action/07-实践-其他/36-身份证识别预处理/1.jpg'
path = '/Users/xingoo/PycharmProjects/ml-in-action/07-实践-其他/36-身份证识别预处理/3.jpeg'
path = '/Users/xingoo/PycharmProjects/ml-in-action/07-实践-其他/36-身份证识别预处理/4.jpg'
path = '/Users/xingoo/Desktop/img/1551944514.549011final.jpg'
#path = '/Users/xingoo/Downloads/Jietu20190304-152230.jpg'

img = cv2.imread(path)
img, scale = resize_im(img, scale=1000, max_scale=1000)
#cv2.imshow('original' , img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow('gray', gray)

# clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(100, 100))
# cl1 = clahe.apply(gray)

# cv2.imshow('clahe', cl1)

gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)
#cv2.imshow("g", gradient)

blurred = cv2.blur(gradient, (3, 3)) # 9*9的核做模糊
(_, thresh) = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
#cv2.imshow("binary", thresh)

closed = cv2.erode(thresh, None, iterations=3)
closed = cv2.dilate(closed, None, iterations=2)
#cv2.imshow("f", closed)

thresh = closed
# 获得水平和竖直方向的像素均值
h, w = thresh.shape
hor_mean = [np.array(thresh[i, :]).mean() for i in range(h)]

hor_mean_reverse = hor_mean.copy()
hor_mean_reverse.reverse()

# 输出函数曲线图，这里输出了水平方向和竖直方向上的像素情况，可以很明显的知道数字的分布
rows, cols = thresh.shape
X = np.arange(0, rows, 1)

plt.subplot(121)
plt.imshow(thresh, 'gray')
plt.subplot(122)
plt.plot(hor_mean_reverse, X)
plt.show()

thresh_value = 10
thresh_w = 30
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



h, w, _ = img.shape
for begin, end in buffer:
    begin -= 10
    end += 10
    cv2.line(img, (0, begin), (w - 1, begin), (0, 255, 0), 1)  # 5
    cv2.line(img, (0, end), (w - 1, end), (0, 255, 0), 1)  # 5

cv2.imshow('r', img)

card_begin, card_end = buffer[-1]
new_img = img[card_begin-10:card_end+10, :, :]
cv2.imshow('card', new_img)

new_thresh = thresh[card_begin-10:card_end+10, :]
cv2.imshow('new_thresh', new_thresh)
#cv2.waitKey(0)

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

h, w, _ = img.shape
for begin, end in buffer:
    # begin -= 5
    # end += 5
    cv2.line(img, (begin, 0), (begin, h-1), (255, 255, 0), 1)  # 5
    cv2.line(img, (end, 0), (end, h-1), (255, 255, 0), 1)  # 5

    cv2.rectangle(img, (begin-5, card_begin), (end+5, card_end), (255, 0, 255), 3)

cv2.imshow('r', img)
cv2.waitKey(0)