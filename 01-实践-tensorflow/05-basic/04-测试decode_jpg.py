import tensorflow as tf
import cv2

img_name = "/Users/xingoo/PycharmProjects/ai/test/3.jpg"
image_jpg = tf.read_file(img_name)
imgage_decode_jpeg = tf.image.decode_jpeg(image_jpg, channels=3, ratio=2, name="decode_jpeg_1")
print(imgage_decode_jpeg.shape)
print(imgage_decode_jpeg.dtype)

sess = tf.Session()
imgage_encode_png = tf.image.encode_png(sess.run(imgage_decode_jpeg), name="encode_png")
print(imgage_encode_png.shape)
print(imgage_encode_png.dtype)

img = tf.image.decode_png(sess.run(imgage_encode_png), name="decode_png")

img = cv2.imshow("img", sess.run(img))
cv2.waitKey(0)