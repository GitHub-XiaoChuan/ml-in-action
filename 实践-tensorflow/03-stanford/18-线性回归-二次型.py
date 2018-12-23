import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

DATA_FILE = 'slr05.xls'

# 第一步， 读取数据
book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

# 第二步，创建输入和输出
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# 第三步，创建变量，并初始化
w = tf.Variable(0.0, name='weights_1')
u = tf.Variable(0.0, name='weights_2')
b = tf.Variable(0.0, name='bias')

# 第四步，构建图
Y_predicted = X * X * w + X * u + b

# 第五步，计算误差
loss = tf.square(Y - Y_predicted, name='loss')

# 第六步，优化
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        for x, y in data:
            sess.run(optimizer, feed_dict={X: x, Y: y})

    w_value, u_value, b_value = sess.run([w, u, b])
    print(w_value)
    print(u_value)
    print(b_value)

plt.plot(data[:, 0], data[:, 1], 'bo', label='Real data')
plt.plot(data[:, 0], data[:, 0] * data[:, 0] * w_value + data[:, 0] * u_value + b_value, 'r', label='Predicted data with squared error')
plt.legend()
plt.show()
