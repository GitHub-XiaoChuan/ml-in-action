import tensorflow as tf

a = tf.Variable(3)
b = a.assign(2*a)

with tf.Session() as sess:
    print(b.eval())
    print(b.eval())
    print(b.eval())