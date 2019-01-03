import cv2
import numpy as np

"""
学习读取，显示，保存视频

https://docs.opencv.org/3.4.2/dd/d43/tutorial_py_video_display.html
"""

cap = cv2.VideoCapture(0)

while 1:
    # ret为true表示正确
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', gray)

    print(cv2.CAP_PROP_FRAME_WIDTH)
    print(cv2.CAP_PROP_FRAME_HEIGHT)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()