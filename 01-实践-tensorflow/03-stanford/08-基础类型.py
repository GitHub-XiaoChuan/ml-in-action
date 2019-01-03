import tensorflow as tf
import numpy as np

t_0 = 19
t_1 = ['a', 'b', 'c']
t_2 = [[True, False, False], [False, False, True], [False, True, False]]
with tf.Session() as sess:
    print(sess.run(tf.zeros_like(t_0)))
    print(sess.run(tf.ones_like(t_0)))

    print(sess.run(tf.zeros_like(t_1)))
    #print(sess.run(tf.ones_like(t_1)))

    print(sess.run(tf.zeros_like(t_2)))
    print(sess.run(tf.ones_like(t_2)))

print(tf.int32 == np.int32)
