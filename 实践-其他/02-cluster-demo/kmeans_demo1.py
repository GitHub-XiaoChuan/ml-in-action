import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state)

'''
n_cluster               类簇的数量
init                    初识中心样本点的初始化方法，k-means++ 或者 random
n_init                  默认跑了十次，会选择其中最好的那次作为结果
max_iter                迭代次数，默认300
tol                     不知道!!!
precompute_distances    不知道!!!
verbose                 不知道!!!
random_state            不知道!!!
copy_x                  不知道!!!
n_jobs                  cpu使用数
algorithm               不知道!!!
'''
kmeans = KMeans(n_clusters=4, random_state=random_state).fit(X)

'''
cluster_centers_    中心点
labels_             标签列表
inertia_            距离中心点的距离（散度）
'''

print(kmeans.cluster_centers_)
print(kmeans.labels_)
print(kmeans.inertia_)

y_pred = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Incorrect Number of Blobs")

plt.show()