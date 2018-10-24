import cv2
import numpy as np

# 创建黑色图片
img = np.zeros((200, 200), dtype=np.uint8)
# 中心创建50宽度和高度的正方形
img[50:150, 50:150] = 255

# 进行二值化
ret, thresh = cv2.threshold(img, 127, 255, 0)

# 三个参数：图像、层次类型、轮廓逼近方法
# 三个返回：修改后的图像，图像的轮廓，对应的层次
# 会修改原图
image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img = cv2.drawContours(color, contours, -1, (0, 255, 0), 2)

cv2.imshow("contours", color)
cv2.waitKey(0)