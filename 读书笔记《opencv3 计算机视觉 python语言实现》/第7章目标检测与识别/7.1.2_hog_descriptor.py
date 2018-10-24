import cv2
import numpy as np
"""
效果真差
"""
def is_inside(o, i):
    ox, oy, ow, oh = o
    ix, iy, iw, ih = i

    return ox > ix and oy > iy \
        and ow + ox < iw + ix  \
        and oh + oy < ih + iy

def draw_person(image, person):
    x, y, w, h = person
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 2)

img = cv2.imread("1.jpeg")
hog = cv2.HOGDescriptor()
# 默认人的HOG
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

found, w = hog.detectMultiScale(img)

found_filtered = []
for ri, r in enumerate(found):
    for qi, q in enumerate(found):
        if ri != qi and is_inside(r, q):
            break
        else:
            found_filtered.append(r)

for person in found_filtered:
    draw_person(img, person)

cv2.imshow('detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()