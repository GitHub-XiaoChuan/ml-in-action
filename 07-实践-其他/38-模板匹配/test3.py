import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np

for file in glob.glob('/Users/xingoo/Documents/dataset/身份证图片/*.jpg'):
    img = cv2.imread(file)
    h, w, _ = img.shape
    scale = 1000/max(h, w)
    img = cv2.resize(img, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度图像
    # edge = cv2.Canny(img, 100, 200)
    # cv2.imshow('res', edge)
    # cv2.waitKey()
    #ttt = cv2.adaptiveThreshold(img, maxValue=150, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C)
    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
    titles = ['img', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
    for i in range(6):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


    #cv2.imshow('ttt', ttt)
    #cv2.waitKey(0)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度图像
    # # open to see how to use: cv2.Canny
    # # http://blog.csdn.net/on2way/article/details/46851451
    # edges = cv2.Canny(gray, 100, 200)
    # plt.subplot(121)
    # plt.imshow(edges, 'gray')
    # plt.xticks([]), plt.yticks([])
    # # hough transform
    # print('before')
    # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=60, maxLineGap=10)
    # print('result')
    # if lines is not None:
    #     lines1 = lines[:, 0, :]  # 提取为二维
    #     for x1, y1, x2, y2 in lines1[:]:
    #         cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
    #
    # plt.subplot(122)
    # plt.imshow(img, )
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()

