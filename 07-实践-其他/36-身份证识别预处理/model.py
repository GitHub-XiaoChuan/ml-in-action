import glob
import os
import cv2
import numpy as np
import tensorflow as tf

path = '/Users/xingoo/Documents/dataset/单字符数据集2'


def generator(image_path, batchsize=32):
    images = glob.glob(os.path.join(image_path, '*.jpg'))

    while 1:
        indexes = np.array(range(0, len(images)))
        np.random.shuffle(indexes)

        inputs = []
        outputs = []

        for index in indexes:
            # 输入数据格式化+标准化
            img = cv2.imread(images[index])
            h, w, _ = img.shape
            max_len = max(h, w)
            image = np.zeros((max_len, max_len, 3))
            image[0:h, 0:w, :] = img
            scale = 32 / max_len
            resize_img = cv2.resize(image, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

            resize_img = resize_img / 255.0 - 0.5
            inputs.append(resize_img)

            # 输出数据标准化
            label = images[index].split('/')[-1].split('_')[0]
            label_vec = np.zeros((11), dtype=np.float32)
            if label != 'X':
                label_vec[int(label)] = 1
            else:
                label_vec[10] = 1
            outputs.append([[label_vec]])

            if len(inputs) == batchsize:
                # print(np.array(inputs).shape)
                # print(np.array(outputs).shape)
                yield np.array(inputs), np.array(outputs)
                inputs = []
                outputs = []


# input = tf.keras.Input(shape=(32, 32), name='input')
# x = tf.keras.layers.Flatten(name='flattern')(input)
# x = tf.keras.layers.Dense(512, activation=tf.nn.relu, name='dense1')(x)
# x = tf.keras.layers.Dropout(0.2, name='dropout')(x)
# output = tf.keras.layers.Dense(11,  activation=tf.nn.softmax, name='dense2')(x)
# model = tf.keras.Model(inputs=input, outputs=output)
# model.summary()

alpha = 1.0
depth_multiplier = 1
# base_model = VGG16(weights=None, include_top=False, input_shape=input)
# base_model.load_weights('../ctpn/pre_trained/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
MobileNet = tf.keras.applications.MobileNet

base_model = MobileNet(weights=None, include_top=False, input_shape=(32, 32, 3))
base_model.load_weights('/Users/xingoo/PycharmProjects/ocr/model/ctpn/pre_trained/mobilenet_1_0_224_tf_no_top.h5')
# base_model.summary()

input = base_model.input
base_output = base_model.get_layer('conv_pw_13_relu').output
output = tf.keras.layers.Dense(11, activation=tf.nn.softmax)(base_output)

model = tf.keras.Model(input, output)
model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

lr_schedule = lambda epoch: 0.0005 * 0.4 ** epoch
learning_rate = np.array([lr_schedule(i) for i in range(50)])
# 学习率的调度器
changelr = tf.keras.callbacks.LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))

earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=1)

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='./checkpoints/idcard-{epoch:02d}.h5', save_weights_only=True)

total = len(glob.glob(os.path.join(path, '*.jpg')))

model.fit_generator(generator(path, 32),
                    steps_per_epoch=total // 32,
                    epochs=10,
                    callbacks=[changelr, checkpoint]
                    )

images = glob.glob('/Users/xingoo/Documents/dataset/单字符样本/*.jpg')
right = 0
for image_path in images:
    img = cv2.imread(image_path, 0)
    h, w = img.shape

    max_len = max(h, w)

    scale = 32 / max_len

    image = np.zeros((max_len, max_len))
    image[0:h, 0:w] = img

    image = cv2.resize(image, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    image = image / 255.0 - 0.5

    input = np.expand_dims(image, axis=0)

    result = model.predict(input).argmax()

    r = 'X'
    if result != 10:
        r = str(result)

    label = image_path.split('/')[-1].split('_')[0]

    print('%s -> %s' % (r, label))
    if label == r:
        right += 1

print('正确率 %s' % str(1 / len(images)))
