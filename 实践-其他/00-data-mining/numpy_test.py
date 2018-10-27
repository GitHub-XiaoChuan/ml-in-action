import numpy as np

# 参考中文文章：https://www.jianshu.com/p/83c8ef18a1e8
# 参考官方文档：https://docs.scipy.org/doc/numpy/user/quickstart.html

np_arr = np.array([[1,2,3],[4,5,6],[7,8,9]])

# 基本方法
print(type(np_arr))
print(np_arr.shape)
print(np_arr.size)
print(np_arr.ndim)
print(np_arr.dtype)

# 创建
array_one = np.ones([3,3])
print(array_one)

array_zero = np.zeros([3,3])
print(array_zero)

print(np.random.rand(3,3)) # 指定形状
print(np.random.uniform(0,100)) # 指定范围的一个数
print(np.random.randint(0,100)) # 指定范围的一个整数
print(np.random.normal(1.75, 0.1, (3,3))) # 正态分布

# 索引 切片
arr = np.random.rand(6,6)
print(arr)

print(arr[1:3,2:4])
print(arr.reshape(2,-1))

# 条件计算
print(arr>0.5)
print(np.where(arr<0.5,0,1)) # 小于0.5的替换成0，大于0.5的替换成1

# 指定轴的最大值和最小值，平均值
print(np.amax(arr, axis=0)) # 每一列的最大值
print(np.amax(arr, axis=1)) # 每一行的最大值
print(np.amin(arr, axis=0)) # 每一列的最小值
print(np.amin(arr, axis=1)) # 每一行的最小值
print(np.mean(arr, axis=0)) # 每一列的平均值
print(np.mean(arr, axis=1)) # 每一行的平均值
print(np.std(arr, axis=0))  # 每一列的方差
print(np.std(arr, axis=1))  # 每一行的方差

# 数组运算 加减乘除 都一样
np_arr[:,0] = np_arr[:,0]+5
print(np_arr)

# 矩阵运算 点乘
arr1 = np.array([[2,2],[2,2]])
arr2 = np.array([2,3])
print(np.dot(arr2, arr1))

# 矩阵拼接 垂直拼接
arr3 = np.array([[3],[3]])
print(np.vstack((arr1, arr2)))
print(np.hstack((arr1, arr3)))

# 读取数据
print(np.genfromtxt("numpy.txt", delimiter=","))