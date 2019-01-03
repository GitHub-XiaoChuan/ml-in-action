import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('1.jpeg', 0)
canny = cv2.Canny(img, 20, 150)

plt.subplot(1, 2, 1)
plt.imshow(img, 'gray')
plt.subplot(1, 2, 2)
plt.imshow(canny, 'gray')
plt.show()


# def nothing(x):
#     pass
# cv2.namedWindow('res')
# cv2.createTrackbar('min','res',0,25,nothing)
# cv2.createTrackbar('max','res',0,25,nothing)
# while(1):
#     if cv2.waitKey(1)&0xFF==27:
#         break
#     maxVal=cv2.getTrackbarPos('max','res')
#     minVal=cv2.getTrackbarPos('min','res')
#     canny=cv2.Canny(img,10*minVal,10*maxVal)
#     cv2.imshow('res',canny)
# cv2.destroyAllWindows()