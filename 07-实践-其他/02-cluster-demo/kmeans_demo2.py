import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 肘点法选择k值

n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state)

distances = []

for iter in np.arange(2,10):
    kmeans = KMeans(n_clusters=iter, random_state=random_state).fit(X)
    distances.append(kmeans.inertia_)

plt.plot(np.arange(2,10),distances)
plt.ylabel('loss distance')
plt.show()