import tensorflow as tf

a1 = tf.Variable(1)
a2 = tf.Variable(2)
update_op = tf.assign(a1, 10)
add = tf.add(a1, a2)

a1_2 = tf.Variable(-1)
a2_2 = tf.Variable(-2)
update_op_2 = tf.assign(a1_2, -10)
# 通过control_dependencies强制添加操作依赖
# 使得使用a1_2之前必须要先使用update_op_2
with tf.control_dependencies([update_op_2]):
    add_2 = tf.add(a1_2, a2_2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ans_1, ans_2 = sess.run([add, add_2])
    print(ans_1)
    print(ans_2)