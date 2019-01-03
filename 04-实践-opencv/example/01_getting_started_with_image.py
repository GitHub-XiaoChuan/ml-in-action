import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
学习基本的图像操作，比如读取、显示、保存
可以基于opencv显示，也可以使用matplotlib显示

https://docs.opencv.org/3.4.2/dc/d2e/tutorial_py_image_display.html
"""

# 读取图片

# cv2.IMREAD_COLOR      # 彩色图片，忽略透明度，默认就是这种处理方式，实际的值为1
# cv2.IMREAD_GRAYSCALE  # 读取灰度图像， 实际的值为0
# cv2.IMREAD_UNCHANGED  # 读取原始图像，包含alpha透明度， 实际的值为-1
img = cv2.imread('1.jpeg', 0)
print(img)

# 如果路径是错的，不会报错，只是img结果为None
img2 = cv2.imread('2.jpeg')
print(img2)

# 显示图片
cv2.imshow('image', img) # 窗口自动适应图片大小
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存图片
cv2.imwrite('test.jpeg', img)

# 使用matplotlib显示图片
plt.imshow(img, 'gray')
plt.show()

"""
PS：
在opencv中是BGR模式，但是在matplotlib中是RGB模式，因此有可能图片颜色会发生错乱。
"""