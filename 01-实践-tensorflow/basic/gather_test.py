import tensorflow as tf
import numpy as np

"""
基于给定的下标序列生成目标序列
"""

a = np.array([0, 10,20,30,40,50,60,70,80])
sess = tf.Session()

# [30 50 20 10]
print(sess.run(tf.gather(a, [3,5,2,1])))