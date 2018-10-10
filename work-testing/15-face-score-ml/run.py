# coding=utf8
import cv2
import dlib
import math
import numpy
import itertools
from sklearn.externals import joblib
from sklearn import decomposition

PREDICTOR_PATH = "./data/shape_predictor_68_face_landmarks.dat"

# 1.使用dlib自带的frontal_face_detector作为我们的人脸提取器
detector = dlib.get_frontal_face_detector()

# 2.使用官方提供的模型构建特征提取器
predictor = dlib.shape_predictor(PREDICTOR_PATH)


def getLandmark(im):
    # 3.使用detector进行人脸检测 rects为返回的结果
    rects = detector(im, 1)

    # 4.输出人脸数，dets的元素个数即为脸的个数
    if len(rects) >= 1:
        print("{} faces detected".format(len(rects)))

    if len(rects) == 0:
        print("no face")

    # f = open('./data/landmarks.txt', 'w')
    lm = []
    for i in range(len(rects)):

        # 5.使用predictor进行人脸关键点识别
        landmarks = numpy.matrix([[p.x, p.y] for p in predictor(im, rects[i]).parts()])
        im = im.copy()

        # 使用enumerate 函数遍历序列中的元素以及它们的下标
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])

            lm.append(point[0, 0])
            lm.append(point[0, 1])

            # cv2.putText(im,str(idx),pos,
            # fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            # fontScale=0.4,

            # color=(0,0,255))
            # 6.绘制特征点
            cv2.circle(im, pos, 3, color=(0, 255, 0))

    return im, numpy.array(lm, dtype=numpy.float64).reshape(1,-1)


def facialRatio(points):
    x1 = points[0]
    y1 = points[1]
    x2 = points[2]
    y2 = points[3]
    x3 = points[4]
    y3 = points[5]
    x4 = points[6]
    y4 = points[7]

    dist1 = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    dist2 = math.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2)

    ratio = dist1 / dist2

    return ratio


def generateFeatures(pointIndices1, pointIndices2, pointIndices3, pointIndices4, allLandmarkCoordinates):
    size = allLandmarkCoordinates.shape
    allFeatures = numpy.zeros((size[0], len(pointIndices1)))
    for x in range(0, size[0]):
        landmarkCoordinates = allLandmarkCoordinates[x, :]
        ratios = []
        for i in range(0, len(pointIndices1)):
            x1 = landmarkCoordinates[2 * (pointIndices1[i] - 1)]
            y1 = landmarkCoordinates[2 * pointIndices1[i] - 1]
            x2 = landmarkCoordinates[2 * (pointIndices2[i] - 1)]
            y2 = landmarkCoordinates[2 * pointIndices2[i] - 1]

            x3 = landmarkCoordinates[2 * (pointIndices3[i] - 1)]
            y3 = landmarkCoordinates[2 * pointIndices3[i] - 1]
            x4 = landmarkCoordinates[2 * (pointIndices4[i] - 1)]
            y4 = landmarkCoordinates[2 * pointIndices4[i] - 1]

            points = [x1, y1, x2, y2, x3, y3, x4, y4]
            ratios.append(facialRatio(points))
        allFeatures[x, :] = numpy.asarray(ratios)
    return allFeatures


def generateAllFeatures(allLandmarkCoordinates):
    a = [18, 22, 23, 27, 37, 40, 43, 46, 28, 32, 34, 36, 5, 9, 13, 49, 55, 52, 58]
    combinations = itertools.combinations(a, 4)
    i = 0
    pointIndices1 = []
    pointIndices2 = []
    pointIndices3 = []
    pointIndices4 = []

    for combination in combinations:
        pointIndices1.append(combination[0])
        pointIndices2.append(combination[1])
        pointIndices3.append(combination[2])
        pointIndices4.append(combination[3])
        i = i + 1
        pointIndices1.append(combination[0])
        pointIndices2.append(combination[2])
        pointIndices3.append(combination[1])
        pointIndices4.append(combination[3])
        i = i + 1
        pointIndices1.append(combination[0])
        pointIndices2.append(combination[3])
        pointIndices3.append(combination[1])
        pointIndices4.append(combination[2])
        i = i + 1

    return generateFeatures(pointIndices1, pointIndices2, pointIndices3, pointIndices4, allLandmarkCoordinates)


if __name__ == '__main__':
    clf = joblib.load('model/my_face_rating.pkl')

    img, lanmarks = getLandmark(cv2.imread('./image/4.jpg'))
    # cv2.imshow('2', img)
    # cv2.waitKey(0)

    my_features = generateAllFeatures(lanmarks)
    print(my_features)

    features = numpy.loadtxt('data/features_ALL.txt', delimiter=',')

    pca = decomposition.PCA(n_components=20)
    pca.fit(features)

    features_test = pca.transform(my_features)
    print(features_test)
    print(clf.predict(features_test))
