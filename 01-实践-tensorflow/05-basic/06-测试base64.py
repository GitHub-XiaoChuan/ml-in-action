import tensorflow as tf
import base64

f = open("/Users/xingoo/PycharmProjects/ai/test/3.jpg", 'rb')
file_content = f.read()
base64str = str(base64.urlsafe_b64encode(file_content), encoding='utf-8')


base64_tensor = tf.convert_to_tensor(base64str, dtype=tf.string)
print(base64_tensor)
img_str = tf.decode_base64(base64_tensor)
#得到（width, height, channel）的图像tensor
img = tf.image.decode_image(img_str, channels=3)
with tf.Session() as sess:
    img_value = sess.run([img])[0] #得到numpy array类型的数据
    print(img_value.shape)