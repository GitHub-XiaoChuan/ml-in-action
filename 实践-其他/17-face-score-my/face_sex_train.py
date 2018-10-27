from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from keras.utils import np_utils
from keras.callbacks import TensorBoard
import os
import numpy as np

def load_dataset(filedir):
    """
    读取数据

    名字为：886_M_5.966666.jpg

    :param filedir:
    :return:
    """
    image_data_list = []
    label = []
    train_image_list = os.listdir(filedir + '/train')
    for img in train_image_list:
        url = os.path.join(filedir + '/train/' + img)
        image = load_img(url, target_size=(128, 128))
        image_data_list.append(img_to_array(image))

        # 解析分值
        sex = img.split("_")[1]
        if sex == "M":
            sex = 0
        else:
            sex = 1
        label.append(sex)

    img_data = np.array(image_data_list)
    img_data = img_data.astype('float32')
    img_data /= 255
    return img_data, label

def make_network():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(128, 128, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    return model


num_threads = os.environ.get('OMP_NUM_THREADS')
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)

if num_threads:
    tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
else:
    tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

train_x, train_y = load_dataset('data')
train_y = np_utils.to_categorical(train_y, 2)
model = make_network()

model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
hist = model.fit(train_x, train_y, batch_size=32, epochs=100, verbose=1)

model.evaluate(train_x,train_y)

model.save('faceSex.h5')
