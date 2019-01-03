import tensorflow as tf

a = tf.placeholder(tf.float32, shape=[3])
b = tf.constant([5, 5, 5], tf.float32)
c = a + b

a1 = tf.add(2, 5)
b1 = tf.multiply(a1, 3)

with tf.Session() as sess:
    print(sess.run(c, {a: [1, 2, 3]}))

    replace_dict = {a1: 15}

    print(sess.run(b1, feed_dict=replace_dict))