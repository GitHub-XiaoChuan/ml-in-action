import numpy as np
import argparse
import cv2

image = cv2.imread('9.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)

thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]

coords = np.column_stack(np.where(thresh > 0))
angle = cv2.minAreaRect(coords)[-1]

if angle < -45:
    angle = -(90+angle)
else:
    angle = -angle

(h, w) = image.shape[:2]
center = (w//2, h//2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(image, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle), (10, 30), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.7, (0,0,255), 2)
cv2.imshow("input", image)
cv2.imshow("gray", gray)
cv2.imshow("thresh", thresh)
cv2.imshow("rotated",rotated)
cv2.waitKey(0)
cv2.waitKey(0)