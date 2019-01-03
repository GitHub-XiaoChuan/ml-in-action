import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics

"""
1 随机K个中心店
2 遍历样本分配到最近的中心上
3 重新计算聚类的中心
4 重复2、3，直到聚类没有什么变化
"""

digits_train = pd.read_csv('optdigits.tra', header=None)
digits_test = pd.read_csv('optdigits.tes', header=None)

x_train = digits_train[np.arange(64)]
y_train = digits_train[64]

x_test = digits_test[np.arange(64)]
y_test = digits_test[64]

kmeans = KMeans(n_clusters=10)
kmeans.fit(x_train)

y_pred = kmeans.predict(x_test)

"""
如果数据本身有类别信息，那么可以使用ARI，Adjusted Rand Index，计算准确性。
"""
print(metrics.adjusted_rand_score(y_test, y_pred))

"""
如果没有类别信息，可以使用轮廓系数进行衡量。
"""