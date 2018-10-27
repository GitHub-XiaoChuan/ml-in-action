import numpy as np
import cv2

# https://blog.csdn.net/liuuze5/article/details/49867317

def main():
    # load the image and convert it to grayscale
    # 加载图像，并且转成灰度图像
    # 彩色图片大部分是RGB的，在做图像处理时，需要转换成灰度图、二值图、HSV、HSI等格式
    image = cv2.imread("/Users/xingoo/PycharmProjects/ml-in-action/work-testing/11-imagerotate/9.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("灰度图", gray)
    cv2.waitKey(0)

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
    cv2.waitKey(0)

    # blur and threshold the image
    # 去噪
    blurred = cv2.blur(gradient, (9, 9)) # 9*9的核做模糊
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY) # 进行二值化

    cv2.imshow("二值化", thresh)
    cv2.waitKey(0)

    # construct a closing kernel and apply it to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # perform a series of erosions and dilations
    # 4次腐蚀 4次膨胀
    closed = cv2.erode(closed, None, iterations = 4)
    closed = cv2.dilate(closed, None, iterations = 4)

    cv2.imshow("移除噪声",closed)
    cv2.waitKey(0)

    # find the contours in the thresholded image, then sort the contours
    # by their area, keeping only the largest one
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

if __name__ == '__main__':
    main()