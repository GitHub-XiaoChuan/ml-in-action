from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np


model = load_model('faceRank.h5')

def load_image(img_url):
    image = load_img(img_url,target_size=(128,128))
    image = img_to_array(image)
    image /= 255
    image = np.expand_dims(image,axis=0)
    return image

image = load_image('data/train/2-112.jpg')
#image = load_image('data/train/9-110.jpg')
p = model.predict_classes(image)
print(p)