import cv2

image = cv2.imread('test.jpeg', 0)
imageVar = cv2.Laplacian(image, cv2.CV_64F).var()
print(imageVar)

image = cv2.imread('test2.jpg', 0)
imageVar = cv2.Laplacian(image, cv2.CV_64F).var()
print(imageVar)