import tensorflow as tf
# 给变量命名
a = tf.constant(2, name="a")
b = tf.constant(3, name="b")
x = tf.add(a, b, name="add")

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs2', sess.graph)
    print(sess.run(x))
writer.close()