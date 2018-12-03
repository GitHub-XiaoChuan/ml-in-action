from tensorflow.examples.tutorials.mnist import input_data
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)