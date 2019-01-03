import time
import numpy as np
import tensorflow as tf
import basic.ptb.reader as reader


class PTBInput(object):
    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        # lstm展开的步数
        self.num_steps = num_steps = config.num_steps
        # 每个epoch多少轮训练
        self.epoch_size = ((len(data)//batch_size)-1)//num_steps
        self.input_data, self.targets = reader.ptb_producer(data, batch_size, num_steps, name=name)

class PTBModel(object):
    def __init__(self, is_training, config, input_):
        """

        :param is_training: 训练标记
        :param config: 配置参数
        :param input_: 输入实例
        """
        self._input = input_

        batch_size = input_.batch_size
        num_steps = input_.num_steps
        # lstm节点数
        size = config.hidden_size
        # 词汇表大小
        vocab_size = config.vocab_size

        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)

        # 配置默认的LSTM单元
        attn_cell = lstm_cell()

        # 如果是训练模式，则增加dropout
        if is_training and config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=config.keep_prob)

        # 对cell进行堆叠
        cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)
        self._initial_state = cell.zero_state(batch_size, tf.float32)

        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [vocab_size, size], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding. input_.input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

