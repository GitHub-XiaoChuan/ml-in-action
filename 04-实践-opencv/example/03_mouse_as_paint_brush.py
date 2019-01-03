import cv2
import numpy as np

"""
学习鼠标事件

mousecallback

https://docs.opencv.org/3.4.2/db/d5b/tutorial_py_mouse_handling.html
"""

# 'EVENT_FLAG_ALTKEY',
# 'EVENT_FLAG_CTRLKEY',
# 'EVENT_FLAG_LBUTTON',
# 'EVENT_FLAG_MBUTTON',
# 'EVENT_FLAG_RBUTTON',
# 'EVENT_FLAG_SHIFTKEY',
# 'EVENT_LBUTTONDBLCLK',
# 'EVENT_LBUTTONDOWN',
# 'EVENT_LBUTTONUP',
# 'EVENT_MBUTTONDBLCLK',
# 'EVENT_MBUTTONDOWN',
# 'EVENT_MBUTTONUP',
# 'EVENT_MOUSEHWHEEL',
# 'EVENT_MOUSEMOVE',
# 'EVENT_MOUSEWHEEL',
# 'EVENT_RBUTTONDBLCLK',
# 'EVENT_RBUTTONDOWN',
# 'EVENT_RBUTTONUP'
events = [i for i in dir(cv2) if 'EVENT' in i]
print(events)


def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img, (x, y), 100, (255, 0, 0), -1)


img = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

while(1):
    cv2.imshow('image', img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()