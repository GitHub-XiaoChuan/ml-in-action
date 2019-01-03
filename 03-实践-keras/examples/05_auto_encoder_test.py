import numpy as np
from keras.layers import Dense, Input
from keras.models import Model
from matplotlib import pyplot as plt

from basic.data import mnist

encoding_dim = 2
input_img = Input(shape=(784,))
# 编码层
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(10, activation='relu')(encoded)
encoder_output = Dense(encoding_dim)(encoded)

# 解码层
decoded = Dense(10, activation='relu')(encoder_output)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='tanh')(decoded)

# 构建自编码模型
autoencoder = Model(inputs=input_img, outputs=decoded)

# 构建编码模型
encoder = Model(inputs=input_img, outputs=encoder_output)

# autoencoder.compile(optimizer='adam', loss='mse')


encoder.load_weights('/Users/xingoo/PycharmProjects/ml-in-action/learn-keras/model/04_auto_encoder1.h5')
autoencoder.load_weights('/Users/xingoo/PycharmProjects/ml-in-action/learn-keras/model/04_auto_encoder2.h5')

X_test, Y_test = mnist.get_test_data_set(10000, True, False)
x_test = np.array(X_test).astype(bool)   # 转化为黑白图
y_test = np.array(Y_test)

encoded_imgs = encoder.predict(x_test)
print(encoded_imgs)
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test, s=6)
plt.colorbar()
# plt.show()

X_test = np.array(X_test)
pred = np.array(autoencoder.predict(x_test))

index = 3
mnist.printimg(X_test[index])
mnist.printimg(np.where(pred[index] > 0, 1, 0))