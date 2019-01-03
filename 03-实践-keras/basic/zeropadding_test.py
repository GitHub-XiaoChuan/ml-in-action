from keras.models import Model
from keras.layers import Input, ZeroPadding2D
import numpy as np

a = Input(shape=(2,2,1))
# (3,1)表示上下填充3行，左右填充1行
b = ZeroPadding2D((3,1))(a)
model = Model(inputs=a,outputs=b)
model.summary()
c = np.asarray([[[[1],[2]],[[3],[4]]]])
print(model.predict(c))