from keras.models import Model
from keras.layers import Input, Lambda
import numpy as np

a = Input(shape=(1,3))

# lambda表达式

b = Lambda(lambda x: x ** 2, output_shape=(1,3), name='l1')(a)
model = Model(inputs=a, outputs=b)

model.summary()

x_test = np.array([[[1,2,3]],[[2,3,4]]])
print(np.shape(x_test))
result = model.predict(x_test)
print(result)

# 自定义函数，并且传参

def my_func(x, param1):
    return x+param1

c = Lambda(my_func, output_shape=(1,3), arguments={'param1':5},name='l2')(a)
model2 = Model(inputs=a, output=c)

model2.summary()

result2 = model2.predict(x_test)
print(result2)