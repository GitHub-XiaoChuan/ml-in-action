import tensorflow as tf

# 第一个例子，计算两个数的加法
a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a, b)

# 在tensorboard中查看
# 执行命令 tensorboard --logdir=/Users/xingoo/PycharmProjects/ml-in-action/实践-tensorflow/03-stanford/graphs

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs1', sess.graph)
    print(sess.run(x))

writer.close()