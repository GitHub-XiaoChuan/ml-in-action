import cv2
import numpy as np
from PIL import Image
import pytesseract

# 光照干扰
#path = '/Users/xingoo/Desktop/img/1551681101.7306502.jpg'
path = '/Users/xingoo/Desktop/img/1551681020.594172.jpg'
#path = '/Users/xingoo/Desktop/img/1551680929.439567.jpg'

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

def calcuRegion(means, value=10, width=30):

    # 当前宽度
    current_width = 0
    # 当前开始位置
    current_begin = 0

    buffer = []

    for current, mean in enumerate(means):
        if mean > value:
            current_width += 1
        else:
            if current_width > width:
                # print('%d - %d' % (current_begin, current))
                buffer.append((current_begin, current))

            current_begin = current
            current_width = 0

    return buffer

img = cv2.imread(path)
img, scale = resize_im(img, scale=1000, max_scale=1000)
resize_img = img.copy()
# 灰度
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# sobel
gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)

# binary
blurred = cv2.blur(gradient, (3, 3)) # 9*9的核做模糊
(_, thresh) = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)

# 获得水平和竖直方向的像素均值
h, w = thresh.shape
hor_mean = [np.array(thresh[i, :]).mean() for i in range(h)]

# 计算水平切线
hor_regions = calcuRegion(hor_mean)

h, w, _ = img.shape
for begin, end in hor_regions:
    begin -= 10
    end += 10
    cv2.line(img, (0, begin), (w - 1, begin), (0, 255, 0), 1)  # 5
    cv2.line(img, (0, end), (w - 1, end), (0, 255, 0), 1)  # 5

# 提取身份证行
card_begin, card_end = hor_regions[-1]
card_img = resize_img[card_begin-10:card_end+10, :, :]
card_thresh = thresh[card_begin-10:card_end+10, :]

# 获得垂直方向的像素均值
card_h, card_w = card_thresh.shape
ver_means = [np.array(card_thresh[:, i]).mean() for i in range(card_w)]

# 计算垂直方向切线
ver_regsions = calcuRegion(ver_means, value=10, width=8)

for begin, end in ver_regsions:
    # begin -= 5
    # end += 5
    cv2.line(img, (begin, 0), (begin, h-1), (255, 255, 0), 1)  # 5
    cv2.line(img, (end, 0), (end, h-1), (255, 255, 0), 1)  # 5

    cv2.rectangle(img, (begin-5, card_begin), (end+5, card_end), (255, 0, 255), 3)

cv2.imshow('r', img)
cv2.imshow('target', card_img)
cv2.waitKey(0)

pilimg = Image.fromarray(card_img, 'RGB')
text = pytesseract.image_to_string((pilimg), lang='eng')
print(text)
