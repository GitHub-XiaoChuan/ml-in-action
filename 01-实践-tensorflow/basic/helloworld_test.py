import tensorflow as tf
tf.enable_eager_execution()
print(tf.add(1, 2))
hello = tf.constant('Hello')
print(hello.numpy())
