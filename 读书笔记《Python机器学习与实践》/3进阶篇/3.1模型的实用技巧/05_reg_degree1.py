import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


x_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]

regressor = LinearRegression()
regressor.fit(x_train, y_train)

xx = np.linspace(0, 26, 100)
xx = xx.reshape(xx.shape[0], 1)
yy = regressor.predict(xx)

plt.scatter(x_train, y_train)

plt1, = plt.plot(xx, yy, label="Degree=1")

plt.axis([0, 25, 0, 25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price')
plt.legend(handles=[plt1])
plt.show()

print('the squared value of lr is', regressor.score(x_train, y_train))