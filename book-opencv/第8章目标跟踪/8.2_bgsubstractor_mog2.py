import cv2
import numpy as np

cap = cv2.VideoCapture(0)
mog = cv2.createBackgroundSubtractorMOG2()

while True:
    rec, frame = cap.read()
    fgmask = mog.apply(frame)

    cv2.imshow('frame', fgmask)
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()