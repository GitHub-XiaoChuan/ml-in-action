import tensorflow as tf
import base64

f = open("/Users/xingoo/PycharmProjects/ai/test/3.jpg", 'rb')
file_content = f.read()
base64str = str(base64.urlsafe_b64encode(file_content), encoding='utf-8')


input_tensor = tf.convert_to_tensor(base64str, dtype=tf.string)
image_str = tf.decode_base64(input_tensor)
img = tf.image.decode_image(image_str, channels=3)

with tf.Session() as sess:
   print(sess.run(image_str))
   print(sess.run(img))