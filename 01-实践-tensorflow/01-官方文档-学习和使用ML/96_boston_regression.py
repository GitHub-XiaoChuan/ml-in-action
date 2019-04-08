import tensorflow as tf
import numpy as np
import pandas as pd
import ssl
import matplotlib.pyplot as plt
from tensorflow import keras

ssl._create_default_https_context = ssl._create_unverified_context

# 0 获取数据

boston_housing = keras.datasets.boston_housing
(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

print(train_data.shape)
print(test_data.shape)

# 1 数据集探索

"""
13个不同的特征：

https://developers.google.com/machine-learning/fairness-overview/

1 人均犯罪率。
2 占地面积超过 25000 平方英尺的住宅用地所占的比例。
3 非零售商业用地所占的比例（英亩/城镇）。
4 查尔斯河虚拟变量（如果大片土地都临近查尔斯河，则为 1；否则为 0）。
5 一氧化氮浓度（以千万分之一为单位）。
6 每栋住宅的平均房间数。
7 1940 年以前建造的自住房所占比例。
8 到 5 个波士顿就业中心的加权距离。
9 辐射式高速公路的可达性系数。
10 每 10000 美元的全额房产税率。
11 生师比（按城镇统计）。
12 1000 * (Bk - 0.63) ** 2，其中 Bk 是黑人所占的比例（按城镇统计）。
13 较低经济阶层人口所占百分比。
"""
print(train_data[0])

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
df = pd.DataFrame(train_data, columns=column_names)
print(df.head())

print(train_labels[0:10])

# 2 特征处理

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

print(train_data[0])

# 3 创建模型

model = keras.Sequential()
model.add(keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(train_data.shape[1],)))
model.add(keras.layers.Dense(64, activation=tf.nn.relu))
model.add(keras.layers.Dense(1))

model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='mse',
              metrics=['mae'])
model.summary()


# 4 训练模型

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 20 == 0:
            print('')
        else:
            print('.', end='')

EPOCHS = 500
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(train_data,
                    train_labels,
                    epochs=EPOCHS,
                    validation_split=0.2,
                    verbose=0,
                    callbacks=[PrintDot(), early_stop]
                    )

plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Mean Abs Error [1000$]')
plt.plot(history.epoch, np.array(history.history['mean_absolute_error']), label='Train Loss')
plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']), label='Val loss')
plt.legend()
plt.ylim([0, 5])
plt.show()

# 5 验证模型

[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)
print("Testing set Mean Abs Error: %{:7.2f}".format(mae * 1000))

# 6 预测模型

test_predictions = model.predict(test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [1000$]')
plt.ylabel('Predictions [1000$]')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

error = test_predictions - test_labels
plt.hist(error, bins=50)
plt.xlabel("Prediction Error[1000$]")
_ = plt.ylabel("Count")
plt.show()