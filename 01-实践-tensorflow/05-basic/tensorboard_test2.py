import tensorflow as tf
import numpy as np
import ssl
"""
知乎：https://zhuanlan.zhihu.com/p/36946874
"""
ssl._create_default_https_context = ssl._create_unverified_context

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
print(y_train.shape)

writer = tf.summary.FileWriter("tmp2", tf.get_default_graph())


def generator(batch_size):
    index = np.arange(0, x_train.shape[0])
    while True:
        np.random.shuffle(index)

        xs = []
        ys = []

        for i in range(batch_size):
            id = index[i]
            xs.append(x_train[id])
            ys.append(y_train[id])

        yield xs, ys


with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(1000):
        xs, ys = generator(10)
        if i % 100 == 0:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys},
                                           options=run_options, run_metadata=run_metadata)
        else:
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})

writer.close()
