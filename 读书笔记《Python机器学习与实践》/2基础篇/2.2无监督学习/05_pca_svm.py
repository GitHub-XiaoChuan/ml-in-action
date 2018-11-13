import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

digits_train = pd.read_csv('optdigits.tra', header=None)
digits_test = pd.read_csv('optdigits.tes', header=None)

x_train = digits_train[np.arange(64)]
y_train = digits_train[64]

x_test = digits_test[np.arange(64)]
y_test = digits_test[64]

# 原始线性预测
svc = LinearSVC()
svc.fit(x_train, y_train)
y_predict = svc.predict(x_test)

estimator = PCA(n_components=20)
pca_x_train = estimator.fit_transform(x_train)
pca_x_test = estimator.transform(x_test)

pca_svc = LinearSVC()
pca_svc.fit(pca_x_train, y_train)
pca_y_predict = pca_svc.predict(pca_x_test)

print(svc.score(x_test, y_test))
print(classification_report(y_test, y_predict, target_names=np.arange(10).astype(str)))

print(pca_svc.score(pca_x_test, y_test))
print(classification_report(y_test, pca_y_predict, target_names=np.arange(10).astype(str)))