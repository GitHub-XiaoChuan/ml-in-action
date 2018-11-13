from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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

# 使用线性回归
lr = LinearRegression()
lr.fit(x_train, y_train)
lr_y_pred = lr.predict(x_test)

# 使用SGD
sgdr = SGDRegressor()
sgdr.fit(x_train, y_train)
sgdr_y_predict = sgdr.predict(x_test)

print('The R2 ', r2_score(y_test, lr_y_pred))
print('The MSE ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_pred)))
print('The MAE ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_pred)))

print('The R2 ', r2_score(y_test, sgdr_y_predict))
print('The MSE ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_predict)))
print('The MAE ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_predict)))