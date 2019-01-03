import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

# M = np.array([[1, 2], [2, 4]])
# np.linalg.matrix_rank(M, tol=None)

digits_train = pd.read_csv('optdigits.tra', header=None)
digits_test = pd.read_csv('optdigits.tes', header=None)

x_digits = digits_train[np.arange(64)]
y_digits = digits_train[64]

estimator = PCA(n_components=2)
x_pca = estimator.fit_transform(x_digits)

def plot_pca_scatter():
    colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']

    for i in range(len(colors)):
        px = x_pca[:, 0][y_digits.as_matrix() == i]
        py = x_pca[:, 1][y_digits.as_matrix() == i]
        plt.scatter(px, py, c=colors[i])

    plt.legend(np.arange(0, 10).astype(str))
    plt.xlabel('First PC')
    plt.ylabel('Second PC')
    plt.show()

plot_pca_scatter()