import cv2
import numpy as np

# https://blog.csdn.net/newthinker_wei/article/details/45603583

img = cv2.imread('1.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

dst = cv2.cornerHarris(gray, 2, 23, 0.04)

img[dst > 0.01 * dst.max()] = [0, 0, 255]
while True:
    cv2.imshow('corners', img)
    if cv2.waitKey(0) & 0xff == ord("q"):
        break
cv2.destroyAllWindows()