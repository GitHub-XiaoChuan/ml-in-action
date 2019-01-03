import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

x_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]

xx = np.linspace(0, 26, 100)
xx = xx.reshape(xx.shape[0], 1)

# 把 [a,b] 转换成 [1, a, b, a^2, ab, b^2]
ploy2 = PolynomialFeatures(degree=2)
x_train_poly2 = ploy2.fit_transform(x_train)

# 基于多项式的线性回归模型
regressor_poly2 = LinearRegression()
regressor_poly2.fit(x_train_poly2, y_train)

xx_poly2 = ploy2.transform(xx)
yy_poly2 = regressor_poly2.predict(xx_poly2)

# 普通的线性回归
regressor = LinearRegression()
regressor.fit(x_train, y_train)

yy = regressor.predict(xx)


poly4 = PolynomialFeatures(degree=4)
x_train_poly4 = poly4.fit_transform(x_train)

regressor_poly4 = LinearRegression()
regressor_poly4.fit(x_train_poly4, y_train)

xx_poly4 = poly4.transform(xx)
yy_poly4 = regressor_poly4.predict(xx_poly4)

# 绘图展示
plt.scatter(x_train, y_train)
plt1, = plt.plot(xx, yy, label='Degree=1')
plt2, = plt.plot(xx, yy_poly2, label='Degree=2')
plt4, = plt.plot(xx, yy_poly4, label='Degree=4')

plt.axis([0,25,0,25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('price')
plt.legend(handles=[plt1, plt2, plt4])
plt.show()
print(regressor.score(x_train, y_train))
print(regressor_poly2.score(x_train_poly2, y_train))
print(regressor_poly4.score(x_train_poly4, y_train))

x_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]

print(regressor.score(x_test, y_test))
x_test_poly2 = ploy2.transform(x_test)
print(regressor_poly2.score(x_test_poly2, y_test))
x_test_poly4 = poly4.transform(x_test)
print(regressor_poly4.score(x_test_poly4, y_test))


# 使用L1正则化
lasso_poly4 = Lasso()
lasso_poly4.fit(x_train_poly4, y_train)
print(lasso_poly4.score(x_test_poly4, y_test))
print(lasso_poly4.coef_)
print(regressor_poly4.score(x_test_poly4, y_test))
print(regressor_poly4.coef_)

# 使用L2正则化
# 显示参数
print(regressor_poly4.coef_)
print(np.sum(regressor_poly4.coef_ ** 2))

ridge_poly4 = Ridge()
ridge_poly4.fit(x_train_poly4, y_train)
print(ridge_poly4.score(x_test_poly4, y_test))
print(ridge_poly4.coef_)
print(np.sum(ridge_poly4.coef_ ** 2))