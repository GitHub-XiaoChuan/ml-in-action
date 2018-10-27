#coding=utf-8
#python 3.5

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

samples = pd.read_csv("./brand.tg", header=None).values
min_max = MinMaxScaler()
x = min_max.fit_transform(samples[:,[4,8]])

y_pred = KMeans(n_clusters=6, random_state=170).fit_predict(x)

plt.scatter(x[:, 0], x[:, 1], c=y_pred)
plt.title("分布")
plt.xlabel("avg_money")
plt.ylabel("members")
plt.show()