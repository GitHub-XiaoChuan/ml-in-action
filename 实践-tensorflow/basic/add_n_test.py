import tensorflow as tf
import numpy as np

input1 = tf.constant([1., 2., 3.])
input2 = tf.Variable(tf.random_uniform([3]))
output = tf.add_n([input1, input2])

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(input1+input2))
    print(sess.run(output))