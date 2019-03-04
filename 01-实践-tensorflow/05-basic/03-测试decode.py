import tensorflow  as tf
import cv2

img_name = "/Users/xingoo/PycharmProjects/ai/test/3.jpg"

image_encode_jpg = tf.read_file(img_name)
img = tf.image.decode_gif(image_encode_jpg, name='decode_gif')
print(img.shape)
print(img.dtype)