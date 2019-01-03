import tensorflow as tf

# tf.Variable是一个class，但是tf.constant只是一个op


a = tf.Variable(2, name='scalar')
b = tf.Variable([2, 3], name='vector')
c = tf.Variable([[0, 1], [2, 3]], name='matrix')
W = tf.Variable(tf.zeros([784, 10]))
d = tf.Variable(10)
d.assign(100)
assign_op = d.assign(101)
#e = tf.Variable(tf.truncated_normal(10, 2))
# initializer\value\assign\assign_add

# 第一种初始化方法：使用全局初始化进行初始化
#init = tf.global_variables_initializer()

# 第二种初始化方法：初始化一部分的变量
init_ab = tf.variables_initializer([a, b], name='init_ab')

my_var = tf.Variable(2, name='my_var')
my_var_times_two = my_var.assign(2*my_var)

with tf.Session() as sess:
    sess.run(init_ab)
    print(sess.run(a))
    print(sess.run(b))
    # 第二种初始化方法：指定某个变量初始化
    sess.run(W.initializer)
    print(W.eval())

    #sess.run(d.initializer)
    #print(d.eval())

    #sess.run(d.initializer) # 这一步可以不做，assgin_op会触发初始化
    sess.run(assign_op)
    print(d.eval())

    sess.run(my_var.initializer)
    print(sess.run(my_var_times_two))
    print(sess.run(my_var_times_two))
    print(sess.run(my_var_times_two))

    print(sess.run(my_var.assign_add(10)))
    print(sess.run(my_var.assign_sub(2)))