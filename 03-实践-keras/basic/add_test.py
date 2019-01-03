from keras.models import Model
from keras.layers import Input, Add
import numpy as np

a = Input(shape=(3,))
b = Input(shape=(3,))
c = Add()([a,b])
model = Model(inputs=[a,b],outputs=c)
model.summary()