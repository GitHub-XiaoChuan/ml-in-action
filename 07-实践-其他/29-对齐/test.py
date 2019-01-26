import numpy
import cv2


def transformation_from_points(cropImg1, cropImg2):
    corners1 = cv2.goodFeaturesToTrack(cropImg1, 25, 0.01, 10)
    n = int(len(corners1))
    corners1 = corners1.reshape(n, 2)
    corners2 = cv2.goodFeaturesToTrack(cropImg2, 25, 0.01, 10)
    n = int(len(corners2))
    corners2 = corners2.reshape(n, 2)

    points1 = corners1
    points2 = corners2
    sumX1 = 0
    sumY1 = 0
    x, y = points1.shape
    for m in range(x):
        sumX1 = sumX1 + points1[m][0]
        sumY1 = sumY1 + points2[m][1]
    sumX2 = 0
    sumY2 = 0
    x, y = points2.shape
    for n in range(x):
        sumX2 = sumX2 + points2[n][0]
        sumY2 = sumY2 + points2[n][1]
    c1 = int((sumX2 / n) - (sumX1 / m))
    c2 = int((sumY2 / 2) - (sumY1 / m))
    return c1, c2


if __name__ == '__main__':
    ph1 = '/Users/xingoo/PycharmProjects/ml-in-action/07-实践-其他/29-对齐/WechatIMG125.jpeg'
    ph2 = '/Users/xingoo/PycharmProjects/ml-in-action/07-实践-其他/29-对齐/WechatIMG126.jpeg'
    s1 = cv2.imread(ph1, 0)
    s2 = cv2.imread(ph2, 0)
    cropImg1 = s1[int(447):int(767), int(361):int(423)]
    cropImg2 = s2[int(y1):int(y1 + height), int(x1):int(x1 + width)]
    c1, c2 = transformation_from_points(cropImg1, cropImg2)
    y2 = y1 + c1
    x2 = x1 + c2
    cropImg2 = s2[int(y2):int(y2 + height), int(x2):int(x2 + width)]