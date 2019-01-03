import tensorflow as tf

sess = tf.InteractiveSession()
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a*b

print(c.eval())
sess.close()