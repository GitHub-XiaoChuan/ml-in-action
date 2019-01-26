import tensorflow as tf

v1 = tf.get_variable('v1', shape=[1], initializer=tf.constant_initializer(1))
tf.add_to_collection('loss', v1)
v2 = tf.get_variable('v2', shape=[1], initializer=tf.constant_initializer(2))
tf.add_to_collection('loss', v2)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(tf.get_collection('loss'))
    print(sess.run(tf.add_n(tf.get_collection('loss'))))
    print(sess.run(tf.add_n([v1, v2])))
