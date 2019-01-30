import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

# 读取数据

DATA_FILE = 'data/fire_theft.xls'

book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
print(sheet.nrows)
print(sheet.ncols)

data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

print(n_samples)
print(data)

# 创建模型
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

W = tf.Variable(0.0, name='weights')
b = tf.Variable(0.0, name='bias')

Y_predict = X * W + b

loss = tf.square(Y - Y_predict, name='loss')

# 开始训练
opt = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
writer = tf.summary.FileWriter('./graphs')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer.add_graph(sess.graph)
    for i in range(100):
        for x, y in data:
            sess.run(opt, feed_dict={X: x, Y: y})

    w_value, b_value = sess.run([W, b])
    print(w_value)
    print(b_value)

writer.close()

plt_x = data[:, 0]
plt_y = data[:, 1]

plt.scatter(plt_x, plt_y)

px = np.arange(0, 50)
py = px*1.7183813 + 15.789157
plt.plot(px, py, c='r')

plt.show()
