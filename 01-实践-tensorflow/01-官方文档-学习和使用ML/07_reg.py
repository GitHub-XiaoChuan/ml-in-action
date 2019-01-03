import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

NUM_WORDS = 10000

(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)


def multi_hot_sequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0
    return results


train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)

plt.plot(train_data[0])
plt.show()

# 降低正则化：减少模型每层的单元数，降低记忆容量；如果容量优先，则难以与训练数据拟合

baseline_model = keras.Sequential()
baseline_model.add(keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)))
baseline_model.add(keras.layers.Dense(16, activation=tf.nn.relu))
baseline_model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

baseline_model.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy', 'binary_crossentropy'])
baseline_model.summary()
baseline_history = baseline_model.fit(train_data,
                   train_labels,
                   epochs=20,
                   batch_size=512,
                   validation_data=(test_data, test_labels),
                   verbose=2)


smaller_model = keras.Sequential()
smaller_model.add(keras.layers.Dense(4, activation=tf.nn.relu, input_shape=(NUM_WORDS,)))
smaller_model.add(keras.layers.Dense(4, activation=tf.nn.relu))
smaller_model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

smaller_model.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy', 'binary_crossentropy'])
smaller_model.summary()
smaller_history = baseline_model.fit(train_data,
                   train_labels,
                   epochs=20,
                   batch_size=512,
                   validation_data=(test_data, test_labels),
                   verbose=2)


bigger_model = keras.Sequential()
bigger_model.add(keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(NUM_WORDS,)))
bigger_model.add(keras.layers.Dense(512, activation=tf.nn.relu))
bigger_model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

bigger_model.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy', 'binary_crossentropy'])
bigger_model.summary()
bigger_history = baseline_model.fit(train_data,
                   train_labels,
                   epochs=20,
                   batch_size=512,
                   validation_data=(test_data, test_labels),
                   verbose=2)

def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(16, 10))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key], '--', label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(), label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()
    plt.xlim([0, max(history.epoch)])
    plt.show()

plot_history([
    ('baseline', baseline_history),
    ('smaller', smaller_history),
    ('bigger', bigger_history)
])