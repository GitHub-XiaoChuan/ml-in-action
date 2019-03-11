import cv2
import glob
import numpy as np
import tensorflow as tf

MobileNet = tf.keras.applications.MobileNet

base_model = MobileNet(weights=None, include_top=False, input_shape=(32, 32, 3))
base_model.load_weights('/Users/xingoo/PycharmProjects/ocr/model/ctpn/pre_trained/mobilenet_1_0_224_tf_no_top.h5')
# base_model.summary()

input = base_model.input
base_output = base_model.get_layer('conv_pw_13_relu').output
output = tf.keras.layers.Dense(11, activation=tf.nn.softmax)(base_output)

model = tf.keras.Model(input, output)
model.summary()

model.load_weights('/Users/xingoo/PycharmProjects/ml-in-action/07-实践-其他/36-身份证识别预处理/checkpoints/idcard-10.h5')


images = glob.glob('/Users/xingoo/Documents/dataset/单字符样本/*.jpg')
right = 0
for image_path in images:
    img = cv2.imread(image_path)
    h, w, _ = img.shape

    max_len = max(h, w)

    scale = 32 / max_len

    image = np.zeros((max_len, max_len, 3))
    image[0:h, 0:w, :] = img

    image = cv2.resize(image, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    image = image / 255.0 - 0.5

    input = np.expand_dims(image, axis=0)

    result = model.predict(input).argmax(axis=-1)[0][0][0]

    r = 'X'
    if result != 10:
        r = str(result)

    label = image_path.split('/')[-1].split('_')[0]

    print('%s -> %s' % (r, label))
    if label == r:
        right += 1

print('正确率 %s' % str(right / len(images)))