import tensorflow as tf

# 创建0填充的张量
a = tf.zeros([2, 3], tf.int32)
b = tf.zeros_like([[0, 1, 2], [3, 4, 5]])

# 创建1填充的张量
c = tf.ones([2, 3], tf.float32)
d = tf.ones_like([0, 1, 2])

# 自定义填充
e = tf.fill([2, 3], 8)

# 间隔
# 10. 到 30. 等间距的形成4个数
f = tf.linspace(10., 30., 4)
# 3 到 19，从3开始累加3，得到序列
g = tf.range(3, 19, 3)
h = tf.range(5)

# 随机初始化
# 初始化服从指定正态分布的数值
r1 = tf.random_normal([10], mean=0.0, stddev=1.0, dtype=tf.float32)
# 产生截断的正态分布，如果与均值差值超过两倍，就重新生成
r2 = tf.truncated_normal([10])
# 产生low和high之间的均匀分布
r3 = tf.random_uniform([10], minval=-3, maxval=3, dtype=tf.float32)
# 随机打乱
r4 = tf.random_shuffle([1,2,3,4,5])
# 随机裁剪
#r5 = tf.random_crop(r3, [5])
# r6 = tf.multinomial()
# r7 = tf.random_gamma([])

with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(c))
    print(sess.run(d))
    print(sess.run(e))

    print(sess.run(f))
    print(sess.run(g))
    print(sess.run(h))

    print(sess.run(r1))
    print(sess.run(r2))
    print(sess.run(r3))
    print(sess.run(r4))
    #print(sess.run(r5))