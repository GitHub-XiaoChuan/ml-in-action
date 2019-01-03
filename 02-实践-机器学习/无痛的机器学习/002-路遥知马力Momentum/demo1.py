import numpy as np
import matplotlib.pyplot as plt
"""
来自：
参考1：https://zhuanlan.zhihu.com/p/21486826
参考2：https://zhuanlan.zhihu.com/p/22810533
参考3：https://blog.csdn.net/BVL10101111/article/details/72614711

动量：一个已经完成的梯度和步长的组合不会立刻消失，而是以一定的形式衰减
"""

def gd(x_start, step, g):
    x = np.array(x_start, dtype='float64')
    passing_dot = [x.copy()]
    for i in range(50):
        grad = g(x)
        x -= grad * step

        passing_dot.append(x.copy())
        print('[ Epoch {0} ] grad = {1}, x = {2}'.format(i, grad, x))
        if abs(sum(grad)) < 1e-6:
            break
    return x, passing_dot

def momentum(x_start, e, g, a=0.7):
    x = np.array(x_start, dtype='float64')
    passing_dot = [x.copy()]
    v = np.zeros_like(x)
    for i in range(50):
        grad = g(x)
        v = a * v - e * grad
        x += v

        passing_dot.append(x.copy())
        print('[ Epoch {0} ] grad = {1}, x = {2}'.format(i, grad, x))
        if abs(sum(grad)) < 1e-6:
            break
    return x, passing_dot

def nesterov(x_start, step, g, discount=0.7):
    x = np.array(x_start, dtype='float64')
    passing_dot = [x.copy()]
    pre_grad = np.zeros_like(x)
    for i in range(50):
        x_future = x - step*discount*pre_grad
        grad = g(x_future)
        pre_grad = pre_grad*discount + grad
        x -= pre_grad * step

        passing_dot.append(x.copy())
        print('[ Epoch {0} ] grad = {1}, x = {2}'.format(i, grad, x))
        if abs(sum(grad)) < 1e-6:
            break
    return x, passing_dot

def f(x):
    return x[0] * x[0] + 50 * x[1] * x[1]

def g(x):
    return np.array([2 * x[0], 100 * x[1]])

def contour(X,Y,Z, arr = None):
    plt.figure(figsize=(15,7))
    xx = X.flatten()
    yy = Y.flatten()
    zz = Z.flatten()
    plt.contour(X, Y, Z, colors='black')
    plt.plot(0,0,marker='*')
    if arr is not None:
        arr = np.array(arr)
        for i in range(len(arr) - 1):
            plt.plot(arr[i:i+2,0],arr[i:i+2,1])
    plt.show()

#原始图
xi = np.linspace(-200, 200, 1000)
yi = np.linspace(-100, 100, 1000)

X, Y = np.meshgrid(xi, yi)

Z = X*X + 50*Y*Y
contour(X,Y,Z)
#
# res, x_arr = gd([150, 75], 0.016, g)
# contour(X, Y, Z, x_arr)
#
# res, x_arr = gd([150,75], 0.019, g)
# contour(X,Y,Z, x_arr)
#
# res, x_arr = gd([150,75], 0.02, g)
# contour(X,Y,Z, x_arr)

res, x_arr = momentum([150,75], 0.016, g)
contour(X,Y,Z, x_arr)

# res, x_arr = nesterov([150,75], 0.012, g)
# contour(X,Y,Z, x_arr)