import tensorflow as tf
import ssl
import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context

# 每张图片都是28*28的
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 数据调查
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# reshape：有返回值，所谓有返回值，即不对原始多维数组进行修改；
# resize：无返回值，所谓有返回值，即会对原始多维数组进行修改；
x_train = x_train.reshape((-1, 784)).astype(np.float32)
y_train = np.eye(10)[y_train].astype(np.float32)
x_test = x_test.reshape((-1, 784)).astype(np.float32)
y_test = np.eye(10)[y_test].astype(np.float32)

print(x_train.shape)
print(y_train.shape)

sess = tf.InteractiveSession()

# 输入
x = tf.placeholder(tf.float32, [None, 784])
# 参数
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# 输出
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

tf.global_variables_initializer().run()

i = 0
while i < len(x_train):
    batch_xs = x_train[i:i + 100]
    batch_ys = y_train[i:i + 100]

    # TODO 现在类型还是匹配不上
    train_step.run({x: batch_xs, y: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # bool转float, 再求平均值
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(accuracy.eval({x: x_test, y_: y_test}))
    i += 100
