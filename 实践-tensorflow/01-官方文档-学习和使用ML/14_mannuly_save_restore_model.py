import tensorflow as tf
from tensorflow import keras
import ssl
import os

ssl._create_default_https_context = ssl._create_unverified_context

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

# 1 创建模型
def create_model():

    model = keras.models.Sequential()

    model.add(keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784, )))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    model.summary()
    return model

model = create_model()

model.fit(train_images,
          train_labels,
          epochs=10,
          validation_data=(test_images, test_labels)
          )
model.save_weights('train_3/my_cp')
loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

model2 = create_model()
model2.load_weights('train_3/my_cp')
loss2, acc2 = model2.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc2))

model.save('my_model.h5')
new_model = keras.models.load_model('my_model.h5')
new_model.summary()
loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))