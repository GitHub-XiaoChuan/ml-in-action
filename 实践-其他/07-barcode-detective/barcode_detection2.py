import numpy as np
import cv2

# https://blog.csdn.net/dcrmg/article/details/52095508
KERNEL_SIZE = (3,3)
SIGMA = 1.5

def main():
    # 1 转换为灰度图
    image = cv2.imread("shrink2.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("1", gray)
    cv2.waitKey(0)

    # 2 高斯平滑滤波
    gaussian = cv2.GaussianBlur(gray, KERNEL_SIZE, SIGMA)
    cv2.imshow("2", gaussian)
    cv2.waitKey(0)

    # 3 求梯度差
    gradX = cv2.Sobel(gaussian, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gaussian, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    # subtract the y-gradient from the x-gradient
    # Scharr操作之后，我们从x-gradient中减去y-gradient，通过这一步减法操作，最终得到包含高水平梯度和低竖直梯度的图像区域。
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    cv2.imshow("3", gradient)
    cv2.waitKey(0)

    # 4 均值滤波
    blurred = cv2.blur(gradient, (3, 3))
    cv2.imshow("4", blurred)
    cv2.waitKey(0)

    # 5 二值化
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)  # 进行二值化
    cv2.imshow("5", thresh)
    cv2.waitKey(0)

    # 6 闭运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    morpho = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("6", morpho)
    cv2.waitKey(0)

    # 7 腐蚀
    closed = cv2.erode(morpho, None, iterations=4)
    cv2.imshow("7", closed)
    cv2.waitKey(0)

    # 8 膨胀
    closed = cv2.dilate(closed, None, iterations=4)
    cv2.imshow("8", closed)
    cv2.waitKey(0)

    (_, cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))

    # draw a bounding box arounded the detected barcode and display the
    # image
    cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.imwrite("result2.png", image)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()