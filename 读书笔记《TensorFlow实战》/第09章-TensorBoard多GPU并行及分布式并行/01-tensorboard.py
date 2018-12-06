import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

max_steps = 1000
learning_rate = 0.001
dropout = 0.9
# data_dir = "/Users/xingoo/PycharmProjects/ml-in-action/读书笔记《TensorFlow实战》/第九章-TensorBoard多GPU并行及分布式并行/data"
# log_dir = "/Users/xingoo/PycharmProjects/ml-in-action/读书笔记《TensorFlow实战》/第九章-TensorBoard多GPU并行及分布式并行/logs"

mnist = input_data.read_data_sets(data_dir, one_hot=True)
sess = tf.InteractiveSession()

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

with tf.name_scope('input_reshape'):
    image_shape_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shape_input, 10)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)