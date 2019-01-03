import tensorflow as tf

# 执行普通的加法
a = tf.add(3, 5)
print(a)

# 在session中执行a
sess = tf.Session()
print(sess.run(a))
sess.close()

# 在session中执行
with tf.Session() as sess:
    print(sess.run(a))

# 复杂的数据流图计算
x = 2
y = 3
op1 = tf.add(x, y)
op2 = tf.multiply(x, y)
op3 = tf.pow(op2, op1)
with tf.Session() as sess:
    op3 = sess.run(op3)
    print(op3)

# 如果fetch的是之前的节点，不会全部执行
# 下面的useless就不会使用
x = 2
y = 3
add_op = tf.add(x, y)
mul_op = tf.multiply(x, y)
useless = tf.multiply(x, add_op)
pow_op = tf.pow(add_op, mul_op)
with tf.Session() as sess:
    z = sess.run(pow_op)
    print(z)

# 一次性计算多个输出结果
with tf.Session() as sess:
    pow_op, useless = sess.run([pow_op, useless])
    print(pow_op)
    print(useless)
