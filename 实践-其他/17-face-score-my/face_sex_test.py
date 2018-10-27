from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import face_recognition
import cv2

model = load_model('faceSex.h5')

def load_image(img_url):
    origin_img = cv2.imread(img_url)
    face_locations = face_recognition.face_locations(origin_img)

    (top, right, bottom, left) = face_locations[0]
    # 截取图片
    face_image = cv2.resize(origin_img[top:bottom, left:right, :], (128, 128))

    face_image = img_to_array(face_image)
    face_image /= 255
    face_image = np.expand_dims(face_image, axis=0)
    return face_image

image = load_image('data/1.jpeg')
p = model.predict_classes(image)
print("赵丽颖: %d" % p)

image = load_image('data/2.jpeg')
p = model.predict_classes(image)
print("王宝强: %d" % p)

image = load_image('data/3.jpeg')
p = model.predict_classes(image)
print("吴彦祖: %d" % p)

image = load_image('data/4.jpeg')
p = model.predict_classes(image)
print("贾玲: %d" % p)

image = load_image('data/5.jpeg')
p = model.predict_classes(image)
print("赵丽颖: %d" % p)

image = load_image('data/6.jpeg')
p = model.predict_classes(image)
print("李宇春: %d" % p)

image = load_image('data/7.jpeg')
p = model.predict_classes(image)
print("赵丽颖: %d" % p)

image = load_image('data/8.jpeg')
p = model.predict_classes(image)
print("面筋哥: %d" % p)

image = load_image('data/9.jpeg')
p = model.predict_classes(image)
print("如花: %d" % p)

image = load_image('data/10.jpeg')
p = model.predict_classes(image)
print("鹿晗: %d" % p)