import tensorflow as tf

W = tf.Variable(10)
U = tf.Variable(2 * W.initial_value)

with tf.Session() as sess:
    sess.run(W.initializer)
    sess.run(U.initializer)
    print(sess.run(U))