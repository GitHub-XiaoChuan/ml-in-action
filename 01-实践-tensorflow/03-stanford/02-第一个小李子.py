import tensorflow as tf

# 第一个例子，计算两个数的加法
a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a, b)
with tf.Session() as sess:
    print(sess.run(x))




