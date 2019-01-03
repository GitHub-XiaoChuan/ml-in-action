import tensorflow as tf
import numpy as np

sess = tf.Session()

a = np.array([[1,0,0],[0,1,1]])
b = np.array([[3,3,3],[3,3,3]])
c = np.array([[4,4,4],[4,4,4]])

# [[ True False False], [False  True  True]]
print(sess.run(tf.equal(a, 1)))

# 如果where里面只有第一个参数，那么返回的是符合条件的下标
# 如果是二维的就是i和j组成的数组
e = np.array([1,1,0,1,0,1])
# [[0] [1] [3] [5]]
print(sess.run(tf.where(tf.equal(e, 1))))

# [[0 0] [1 1] [1 2]]
print(sess.run(tf.where(tf.equal(a, 1))))

f = np.array([[[1,0,1],[0,0,0]],[[1,0,0],[0,0,0]]])
# [[0 0 0] [0 0 2] [1 0 0]]
print(sess.run(tf.where(tf.equal(f, 1))))


# 如果包含 x和y，那么符合条件就用x填充，不符合条件就用y填充
# [[3 4 4] [4 3 3]]
print(sess.run(tf.where(tf.equal(a, 1), b, c)))