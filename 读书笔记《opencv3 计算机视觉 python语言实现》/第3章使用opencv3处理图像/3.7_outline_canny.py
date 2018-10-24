import cv2
import matplotlib.pyplot as plt

"""
1 高斯滤波器去噪
2 计算梯度
3 边缘使用非最大值抑制（NMS）
4 在检测边缘使用双阈值去除假阳性
5 分析边缘之间的连接，保留真正的边缘，消除不明显边缘
"""

img = cv2.imread('../2.jpeg', 0)
canny = cv2.Canny(img, 30, 50)
plt.subplot(1, 1, 1)
plt.imshow(canny, 'gray')
plt.show()
