import cv2
import face_recognition
from keras.models import load_model, Model
from keras.layers import Input, Activation, add, Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
import numpy as np
import sys
import logging

sys.setrecursionlimit(2 ** 20)
np.random.seed(2 ** 10)


class WideResNet:
    def __init__(self, image_size, depth=16, k=8):
        self._depth = depth
        self._k = k
        self._dropout_probability = 0
        self._weight_decay = 0.0005
        self._use_bias = False
        self._weight_init = "he_normal"

        if K.image_dim_ordering() == "th":
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)

    # Wide residual network http://arxiv.org/abs/1605.07146
    def _wide_basic(self, n_input_plane, n_output_plane, stride):
        def f(net):
            # format of conv_params:
            #               [ [kernel_size=("kernel width", "kernel height"),
            #               strides="(stride_vertical,stride_horizontal)",
            #               padding="same" or "valid"] ]
            # B(3,3): orignal <<basic>> block
            conv_params = [[3, 3, stride, "same"],
                           [3, 3, (1, 1), "same"]]

            n_bottleneck_plane = n_output_plane

            # Residual block
            for i, v in enumerate(conv_params):
                if i == 0:
                    if n_input_plane != n_output_plane:
                        net = BatchNormalization(axis=self._channel_axis)(net)
                        net = Activation("relu")(net)
                        convs = net
                    else:
                        convs = BatchNormalization(axis=self._channel_axis)(net)
                        convs = Activation("relu")(convs)

                    convs = Conv2D(n_bottleneck_plane, kernel_size=(v[0], v[1]),
                                   strides=v[2],
                                   padding=v[3],
                                   kernel_initializer=self._weight_init,
                                   kernel_regularizer=l2(self._weight_decay),
                                   use_bias=self._use_bias)(convs)
                else:
                    convs = BatchNormalization(axis=self._channel_axis)(convs)
                    convs = Activation("relu")(convs)
                    if self._dropout_probability > 0:
                        convs = Dropout(self._dropout_probability)(convs)
                    convs = Conv2D(n_bottleneck_plane, kernel_size=(v[0], v[1]),
                                   strides=v[2],
                                   padding=v[3],
                                   kernel_initializer=self._weight_init,
                                   kernel_regularizer=l2(self._weight_decay),
                                   use_bias=self._use_bias)(convs)

            # Shortcut Connection: identity function or 1x1 convolutional
            #  (depends on difference between input & output shape - this
            #   corresponds to whether we are using the first block in each
            #   group; see _layer() ).
            if n_input_plane != n_output_plane:
                shortcut = Conv2D(n_output_plane, kernel_size=(1, 1),
                                  strides=stride,
                                  padding="same",
                                  kernel_initializer=self._weight_init,
                                  kernel_regularizer=l2(self._weight_decay),
                                  use_bias=self._use_bias)(net)
            else:
                shortcut = net

            return add([convs, shortcut])

        return f

    # "Stacking Residual Units on the same stage"
    def _layer(self, block, n_input_plane, n_output_plane, count, stride):
        def f(net):
            net = block(n_input_plane, n_output_plane, stride)(net)
            for i in range(2, int(count + 1)):
                net = block(n_output_plane, n_output_plane, stride=(1, 1))(net)
            return net

        return f

    #    def create_model(self):
    def __call__(self):
        logging.debug("Creating model...")

        assert ((self._depth - 4) % 6 == 0)
        n = (self._depth - 4) / 6

        inputs = Input(shape=self._input_shape)

        n_stages = [16, 16 * self._k, 32 * self._k, 64 * self._k]

        conv1 = Conv2D(filters=n_stages[0], kernel_size=(3, 3),
                       strides=(1, 1),
                       padding="same",
                       kernel_initializer=self._weight_init,
                       kernel_regularizer=l2(self._weight_decay),
                       use_bias=self._use_bias)(inputs)  # "One conv at the beginning (spatial size: 32x32)"

        # Add wide residual blocks
        block_fn = self._wide_basic
        conv2 = self._layer(block_fn, n_input_plane=n_stages[0], n_output_plane=n_stages[1], count=n, stride=(1, 1))(
            conv1)
        conv3 = self._layer(block_fn, n_input_plane=n_stages[1], n_output_plane=n_stages[2], count=n, stride=(2, 2))(
            conv2)
        conv4 = self._layer(block_fn, n_input_plane=n_stages[2], n_output_plane=n_stages[3], count=n, stride=(2, 2))(
            conv3)
        batch_norm = BatchNormalization(axis=self._channel_axis)(conv4)
        relu = Activation("relu")(batch_norm)

        # Classifier block
        pool = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), padding="same")(relu)
        flatten = Flatten()(pool)
        predictions_g = Dense(units=2, kernel_initializer=self._weight_init, use_bias=self._use_bias,
                              kernel_regularizer=l2(self._weight_decay), activation="softmax",
                              name="pred_gender")(flatten)
        predictions_a = Dense(units=101, kernel_initializer=self._weight_init, use_bias=self._use_bias,
                              kernel_regularizer=l2(self._weight_decay), activation="softmax",
                              name="pred_age")(flatten)
        model = Model(inputs=inputs, outputs=[predictions_g, predictions_a])

        return model


known_face_encodings = []
known_face_names = []


def load_face_encodings(path):
    encoding_lines = []
    with open(path, 'r') as f:
        encoding_lines = f.readlines()

    for encoding_line in encoding_lines:
        arr = encoding_line.strip('\n').split(' ')
        known_face_names.append(arr[0])
        known_face_encodings.append([float(x) for x in arr[1:]])

    None

def face_matches(face_encodings):
    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        else:
            # name = len(known_face_names)+1000
            # known_face_encodings.append(face_encoding)
            # known_face_names.append(name)

            name = 'unknow'

        face_names.append(name)
    return face_names


def main():
    camera = cv2.VideoCapture(0)

    #load_face_encodings('db.txt')
    load_face_encodings('db2.txt')

    print("底库名字总共 %d 个" % len(known_face_names))
    print("底库编码总共 %d 个" % len(known_face_encodings))

    face_locations = []
    face_encodings = []

    while True:
        ret, frame = camera.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)
        rgb_small_frame = small_frame[:, :, ::-1]  # BGR转RGB

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters=10)
        face_names = face_matches(face_encodings)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 5
            right *= 5
            bottom *= 5
            left *= 5

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # image128 = cv2.resize(frame[top:bottom, left:right], (128, 128))
            # images = np.array(image128).reshape(1, 128, 128, 3)

            # score = score_model.predict_classes(images)
            # sex = sex_model.predict_classes(images)

            # image64 = cv2.resize(frame[top:bottom, left:right], (64, 64))
            # images = np.array(image64).reshape(1, 64, 64, 3)
            # results = model.predict(images)
            #
            # predicted_genders = results[0]
            # ages = np.arange(0, 101).reshape(101, 1)
            # predicted_ages = results[1].dot(ages).flatten()
            #
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            # cv2.putText(frame, "%d %s %d %s" % (
            #     score,
            #     "F" if predicted_genders[0][0] > 0.5 else "M",
            #     int(predicted_ages[0])-10,
            #     str(name)
            # ), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            cv2.putText(frame, str(name), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow("video", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


score_model = load_model('faceRank.h5')
# sex_model = load_model('faceSex.h5')

model = WideResNet(64, depth=16, k=8)()
model.load_weights('weights.28-3.73.hdf5')

if __name__ == '__main__':
    main()
