import cv2

def resize(img, scaleFactor):
    return cv2.resize(img,
                      (int(img.shape[1]*(1/scaleFactor)),
                            int(img.shape[0]*(1/scaleFactor))),
                      interpolation=cv2.INTER_AREA
                      )

def pyramid(image, scale=1.5, minSize=(200, 80)):
    yield image

    while True:
        image = resize(image, scale)
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        yield image

if __name__ == '__main__':
    image = cv2.imread('1.jpeg')
    for resized in pyramid(image):
        cv2.imshow('test', resized)
        cv2.waitKey(0)