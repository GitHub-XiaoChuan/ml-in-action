import cv2

def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y+windowSize[1], x:x+windowSize[0]])

if __name__ == '__main__':
    image = cv2.imread('1.jpeg')
    for x, y, img in sliding_window(image, 100, (200, 200)):
        target = image.copy()  
        cv2.rectangle(target, (x, y), (x+200, y+200), (0, 255, 255), 2)

        cv2.imshow('target', target)
        cv2.waitKey(0)