from sklearn.svm import SVR
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

# 使用线性核函数配置
linear_svr = SVR(kernel='linear')
linear_svr.fit(x_train, y_train)
linear_svr_y_predict = linear_svr.predict(x_test)

# 使用多项式核函数配置
ploy_svr = SVR(kernel='poly')
ploy_svr.fit(x_train, y_train)
ploy_svr_y_predict = ploy_svr.predict(x_test)

# 使用径向基核函数配置
rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(x_train, y_train)
rbf_svr_y_predict = rbf_svr.predict(x_test)


print('The R2 ', r2_score(y_test, linear_svr_y_predict))
print('The MSE ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))
print('The MAE ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))


print('The R2 ', r2_score(y_test, ploy_svr_y_predict))
print('The MSE ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(ploy_svr_y_predict)))
print('The MAE ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(ploy_svr_y_predict)))


print('The R2 ', r2_score(y_test, rbf_svr_y_predict))
print('The MSE ', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
print('The MAE ', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))