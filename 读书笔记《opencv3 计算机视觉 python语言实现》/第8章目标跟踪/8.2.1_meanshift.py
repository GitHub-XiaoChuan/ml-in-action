import numpy as np
import cv2

camera = cv2.VideoCapture(0)
# 第一帧用不了
_, frame = camera.read()

track_window = None

while True:
    ret, frame = camera.read()

    if track_window == None:
        x, y, w, h = 450, 450, 250, 250
        track_window = (x, y, w, h)

        roi = frame[x:x + w, y:y + h]
        cv2.imshow('roi', roi)
        cv2.waitKey(0)

        hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((100., 30., 32.)), np.array((180., 120., 255.)))

        """
        calcHist()函数：
        参数解释：
        images 原数组
        channels 通道数
        mask 掩码
        hisSize 直方图数组大小
        ranges 上下界
        hist 输出直方图
        accumulate累计标准比
        """
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    if ret:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)


        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        x, y, w, h = track_window
        img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
        cv2.imshow('img2', frame)
        if cv2.waitKey(1) == ord("q"):
            break
    else:
        break
cv2.destroyAllWindows()
camera.release()
