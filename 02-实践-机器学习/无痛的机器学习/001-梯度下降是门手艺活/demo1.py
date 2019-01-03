import numpy as np
import matplotlib.pyplot as plt

"""
文章来自 https://zhuanlan.zhihu.com/p/21486804

梯度下降其实就是针对函数求解它的导数，在乘以负梯度。
因为梯度包含了方向的关系，负梯度就隐含了X变化的方向。
"""

def gd(x_start, step, g):
    x = x_start

    for i in range(20):
        grad = g(x)
        x -= grad * step
        print('[ Epoch {0} ] grad = {1}, x = {2}'.format(i, grad, x))
        if abs(grad) < 1e-6:
            break
    return x

def f(x):
    return x*x - 2*x + 1

def g(x):
    return 2 * x - 2


x = np.linspace(-5,7,100)
y = f(x)

gd(5,1,g)

plt.plot(x, y)
plt.show()