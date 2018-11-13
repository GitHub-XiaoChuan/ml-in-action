from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


"""
树模型的优点：
1 解决非线性特征问题
2 数值型和类别都可以直接应用
3 具有更高的解释性

缺陷：
1 非线性拟合，容易导致过于复杂，而丧失泛化能力
2 树模型从上到下，会因为细微的改变，而使结果变化很大
3 树模型是个NP问题，借助多个模型，在多次优解中寻找最优解

"""

boston = load_boston()
print(boston.DESCR)

x = boston.data
y = boston.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

print('max is ', np.max(boston.target))
print('min is ', np.min(boston.target))
print('mean is ', np.mean(boston.target))

ss_x = StandardScaler()
ss_y = StandardScaler()

x_train = ss_x.fit_transform(x_train)
x_test = ss_x.transform(x_test)

y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)

dtr = DecisionTreeRegressor()
dtr.fit(x_train, y_train)
dtr_y_predict = dtr.predict(x_test)

print('The R2 ', r2_score(y_test, dtr_y_predict))
print('The MSE ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dtr_y_predict)))
print('The MAE ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dtr_y_predict)))