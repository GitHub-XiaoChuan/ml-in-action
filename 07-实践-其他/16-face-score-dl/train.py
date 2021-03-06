
from __future__ import print_function
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import numpy
import tensorflow as tf


# Network Parameters
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, 128, 128, 3])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 128, 128, 3])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    print(conv1.shape)
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
    print(conv1.shape)
    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    print(conv2.shape)
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)
    print(conv2.shape)
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 3 input, 24 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 24])),
    # 5x5 conv, 24 inputs, 96 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 24, 96])),
    # fully connected, 32*32*96 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([32*32*96, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, 10]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([24])),
    'bc2': tf.Variable(tf.random_normal([96])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([10]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
saver=tf.train.Saver()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    list = os.listdir("./resize_image/")
    #print(list)
    #print(len(list))
    count=0
    while count < 500:
        count = count + 1
        #print('count:',count)
        batch_xs = []
        batch_ys = []
        for image in list:
            id_tag = image.find("-")
            score = image[0:id_tag]
            # print(score)
            img = Image.open("./resize_image/" + image)
            img_ndarray = numpy.asarray(img, dtype='float32')
            img_ndarray = numpy.reshape(img_ndarray, [128, 128, 3])
            # print(img_ndarray.shape)
            batch_x = img_ndarray
            batch_xs.append(batch_x)
            # print(batch_xs)
            batch_y = numpy.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            # print(type(score))
            batch_y[int(score) - 1] = 1
            # print(batch_y)
            batch_y = numpy.reshape(batch_y, [10, ])
            batch_ys.append(batch_y)
        # print(batch_ys)
        batch_xs = numpy.asarray(batch_xs)
        #print(batch_xs.shape)
        batch_ys = numpy.asarray(batch_ys)
        #print(batch_ys.shape)

        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys,
                                       keep_prob: dropout})
        print("count = " + str(count))
        if count % 10 == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_xs,
                                                              y: batch_ys,
                                                              keep_prob: 1.})
            print("count = " + str(count) + ", Minibatch Loss= " + str(loss) + ", Training Accuracy= " + str(acc) )
    print("Optimization Finished!")
    saver.save(sess,"./model/model.ckpt")


