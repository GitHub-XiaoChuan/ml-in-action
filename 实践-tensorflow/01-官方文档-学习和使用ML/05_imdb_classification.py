import matplotlib.pyplot as plt
import tensorflow as tf
import ssl
from tensorflow import keras

ssl._create_default_https_context = ssl._create_unverified_context

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# 1 探索数据
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
print(train_data[0])
# 影评长度不同
print("%d %d" % (len(train_data[0]), len(train_data[1])))

# 整数转换会文本
word_index = imdb.get_word_index()

word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(train_data[0]))

# 2 准备数据
train_data = keras.preprocessing.sequence.pad_sequences(
    train_data,
    value=word_index["<PAD>"],
    padding='post',
    maxlen=256
)

test_data = keras.preprocessing.sequence.pad_sequences(
    test_data,
    value=word_index["<PAD>"],
    padding='post',
    maxlen=256
)

print("%d %d" % (len(train_data[0]), len(train_data[1])))
print(train_data[0])

# 3 构建模型
vocab_size = 10000

model = keras.Sequential()
# 词嵌入
model.add(keras.layers.Embedding(vocab_size, 16))
# 序列维度平均值
model.add(keras.layers.GlobalAveragePooling1D())
# 全连接
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
# sigmoid结果
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 4 创建验证集

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# 5 训练模型

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# 6 评估模型

results = model.evaluate(test_data, test_labels)
print(results)

# 7 准确率和损失变化图
history_dict = history.history
history_dict.keys()

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

# 绘制损失曲线
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# 绘制准确率曲线

plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()