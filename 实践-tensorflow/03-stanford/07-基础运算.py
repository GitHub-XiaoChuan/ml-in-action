import tensorflow as tf

a = tf.constant([3, 6])
b = tf.constant([2, 2])

r1 = tf.add(a, b, name='add')
r2 = tf.add_n([a, b, b], name='add_n')
r3 = tf.multiply(a, b, name='multiply')
#r4 = tf.matmul([2], [3], name='matmul')
r5 = tf.matmul(tf.reshape(a, [1, 2]), tf.reshape(b, [2, 1]), name='matmul2')
r6 = tf.div(a, b)
r7 = tf.mod(a, b)

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)

    print(sess.run(r1))
    print(sess.run(r2))
    print(sess.run(r3))
    #print(sess.run(r4))
    print(sess.run(r5))
    print(sess.run(r6))
    print(sess.run(r7))

writer.close()
