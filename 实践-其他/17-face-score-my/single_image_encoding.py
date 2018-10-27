import cv2
import face_recognition

origin_img = cv2.imread('/Users/xingoo/Desktop/123.jpg')
face_locations = face_recognition.face_locations(origin_img)
(top, right, bottom, left) = face_locations[0]
face_image = cv2.resize(origin_img[top:bottom, left:right, :], (128, 128))

encodings = face_recognition.face_encodings(origin_img[top:bottom, left:right, :], num_jitters=10)
print("1111 " + " ".join([str(x) for x in encodings[0]]))