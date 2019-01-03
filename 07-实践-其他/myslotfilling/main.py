from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.core import Dense, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers import Convolution1D, MaxPooling1D
from sklearn.preprocessing import LabelEncoder

import numpy as np

# https://chsasank.github.io/spoken-language-understanding.html

x = []
y = []
x_set = set()
y_set = set()

with open('data', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        arr_x = []
        arr_y = []
        for str in line.strip().split(" "):
            word,label = str.split(":")
            arr_x.append(word)
            arr_y.append(label)
            x_set.add(word)
            y_set.add(label)

        x.append(arr_x[-4:])
        y.append(arr_y[-4:])

label_encoder = LabelEncoder()

x2index = {v: k for k,v in enumerate(x_set)}
y2index = {v: k for k,v in enumerate(y_set)}

train_x = []
train_y = []

for list in x:
    x_arr = []
    for v in list:
        x_arr.append(x2index[v])
    train_x.append(x_arr)

for list in y:
    y_arr = []
    for v in list:
        y_arr.append(y2index[v])
    train_y.append(y_arr)


print(train_y)
print(train_x)
print(x2index)
print(x_set)
print(y_set)


model = Sequential()
model.add(Embedding(len(x2index),10))
model.add(Dropout(0.10))
model.add(SimpleRNN(4,return_sequences=True))
model.add(TimeDistributed(Dense(len(y2index), activation='softmax')))
model.compile('rmsprop', 'categorical_crossentropy')

print(np.array(train_x).shape)
print(np.array(train_y).shape)

model.fit(train_x, train_y)

# loss = model.train_on_batch(np.array(train_x)[np.newaxis, :],np.array(train_y)[np.newaxis, :])
#print(loss)