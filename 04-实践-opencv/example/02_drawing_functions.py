import cv2
import numpy as np

"""
学习opencv中不同的形状
line、circle、rectanle、ellipse、text

通用参数：
img 图片
color 颜色，BGR模式
thickness，-1表示填充，默认是1
LineType，线型

https://docs.opencv.org/3.4.2/dc/da5/tutorial_py_drawing_functions.html
"""

img = np.zeros((512, 512, 3), np.uint8)

# 起始坐标、终点坐标、颜色、宽度
line = cv2.line(img, (0, 0), (511, 511), (255, 0, 0), 5)
cv2.imshow('line', line)
cv2.waitKey(0)

# 左上角xy，右下角xy，颜色，边框宽度
rec = cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)
cv2.imshow('rect', rec)
cv2.waitKey(0)

# 中心点，xy的长度，旋转的角度，开始的角度，结束的角度，颜色，是否完全填充
ellipse = cv2.ellipse(img, (256, 256), (100, 50), 30, 0, 180, (0, 0, 255), -1)
cv2.imshow('ellipse', ellipse)
cv2.waitKey(0)

pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
print(pts)
# 第三个参数控制是否闭合
poly = cv2.polylines(img, [pts], True, (0, 255, 255))
cv2.imshow('poly', poly)
cv2.waitKey(0)


font = cv2.FONT_HERSHEY_SIMPLEX
# 想写的文本, 左下角的坐标位置，字体，大小，颜色，粗细，线性
cv2.putText(img, 'opencv', (10, 500), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
cv2.imshow('text', img)
cv2.waitKey(0)