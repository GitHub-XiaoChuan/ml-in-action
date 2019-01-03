import numpy as np

t = np.arange(4)
print(t)
print(t.transpose())

t2 = np.arange(16).reshape(4,4)
print(t2)
# 二维相当于是x,y轴，分别对应0 1轴，这里如果是1，0表示两个轴的位置互换
print(t2.transpose(1,0))
print(t2.transpose(0,1))

t3 = np.arange(8).reshape(2,2,2)
print(t3)
# 下标为 a,b,c 相当于变成了c,a,b
print(t3.transpose(2,0,1))