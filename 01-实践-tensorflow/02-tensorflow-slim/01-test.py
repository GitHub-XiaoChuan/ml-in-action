from tensorflow.contrib import slim
import tensorflow as tf

weights = slim.variable('weights',
                        shape=[10, 10, 3 , 3],
                         initializer=tf.truncated_normal_initializer(stddev=0.1),
                         regularizer=slim.l2_regularizer(0.05),
                         device='/CPU:0')

# Model Variables
weights = slim.model_variable('weights',
                              shape=[10, 10, 3 , 3],
                              initializer=tf.truncated_normal_initializer(stddev=0.1),
                              regularizer=slim.l2_regularizer(0.05),
                              device='/CPU:0')
model_variables = slim.get_model_variables()

# Regular variables
my_var = slim.variable('my_var',
                       shape=[20, 1],
                       initializer=tf.zeros_initializer())
regular_variables_and_model_variables = slim.get_variables()