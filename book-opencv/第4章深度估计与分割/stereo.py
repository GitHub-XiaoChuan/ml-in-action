import cv2
import numpy as np

def update(val=0):
    stereo.setBlockSize(cv2.getTrackbarPos('window_size', 'disparity'))
    stereo.setBlockSize(cv2.getTrackbarPos('uniquenessRatio', 'disparity'))
    stereo.setBlockSize(cv2.getTrackbarPos('speckleWindowSize', 'disparity'))
    stereo.setBlockSize(cv2.getTrackbarPos('speckleRange', 'disparity'))
    stereo.setBlockSize(cv2.getTrackbarPos('disp12MaxDiff', 'disparity'))

    print('computing dispatity')
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    cv2.imshow('left', imgL)
    cv2.imshow('disparity', (disp - min_disp)/num_disp)

if __name__ == '__main__':
    window_size = 5
    min_disp = 16
    num_disp = 192 - min_disp
    blockSize = window_size
    uniquenessRatio = 1
    speckleRange = 3
    speckleWindowSize = 3
    disp12MaxDiff = 200
    P1 = 600
    P2 = 2400

    imgL = cv2.imread('left1.jpg')
    imgR = cv2.imread('right1.jpg')

    cv2.namedWindow('disparity')
    cv2.createTrackbar('speckleRange', 'disparity', speckleRange, 50, update)
    cv2.createTrackbar('window_size', 'disparity', window_size, 21, update)
    cv2.createTrackbar('speckleWindowSize', 'disparity', speckleWindowSize, 200, update)
    cv2.createTrackbar('uniquenessRatio', 'disparity', uniquenessRatio, 50, update)
    cv2.createTrackbar('disp12MaxDiff', 'disparity', disp12MaxDiff, 250, update)

    """
    SGBM 是 semiglobal block matching的缩写，是一种计算视差图的算法
    minDisparity 表示最小的视差值
    numDIsparity 表示最大视差值与最小值的差，需要被16整除
    windowSize 匹配块的大小，3-11之间的奇数
    P1 参数控制视差图平滑度
    P2 参数控制是插入平滑度，越大越平滑
    disp12MaxDiff 左右视图最大允许的偏差
    preFilterCap 预过滤的截断值
    uniquenessRatio 
    speckleWindowsize 平滑视差的最大窗口尺寸
    speckleRange 最大视差变化
    """

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        uniquenessRatio=uniquenessRatio,
        speckleRange=speckleRange,
        speckleWindowSize=speckleWindowSize,
        disp12MaxDiff=disp12MaxDiff,
        P1=P1,
        P2=P2
    )
    update()
    cv2.waitKey(0)