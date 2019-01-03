import tensorflow as tf

# 常量存储在图中
a = 2
b = 3
x = tf.add(a, b)
y = tf.multiply(a, b)
useless = tf.multiply(a, b)
z = tf.pow(y, x)

# 创建session分配内存，存储变量
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    z = sess.run(z)
    print(z)
    writer.close()