import tensorflow as tf

# tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=False)

a = tf.constant([1, 3], name="a")
b = tf.constant([[0, 1], [2, 3]], name="b")

x = tf.add(a, b, name="add")
y = tf.multiply(a, b, name="mul")

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    x, y = sess.run([x, y])
    print(x)
    print(y)
    writer.close()
