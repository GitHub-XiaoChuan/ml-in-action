#coding=utf8
import argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn import decomposition
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn import gaussian_process
from sklearn.externals import joblib

#parser = argparse.ArgumentParser()
#parser.add_argument('-model', type=str, default='linear_model')
#parser.add_argument('-featuredim', type=str, default=20)
#parser.add_argument('-inputfeatures', type=str, default='../data/features_ALL.txt')
#parser.add_argument('-labels', type=str, default='../data/ratings.txt')
#args = parser.parse_args()

#replace it by your own folder path

features = np.loadtxt('./data/features_ALL.txt', delimiter=',')
#features = preprocessing.scale(features)
features_train = features[0:-50]
features_test = features[-50:]

pca = decomposition.PCA(n_components=20)
pca.fit(features_train)
features_train = pca.transform(features_train)
features_test = pca.transform(features_test)

ratings = np.loadtxt('./data/ratings.txt', delimiter=',')
#ratings = preprocessing.scale(ratings)
ratings_train = ratings[0:-50]
ratings_test = ratings[-50:]

regr = RandomForestRegressor(
    n_estimators=50,
    max_depth=10,
    min_samples_split=8,
    min_samples_leaf=1,
    random_state=20)
#regr = linear_model.LinearRegression()
#regr = svm.SVR()
#regr.fit(features_train, ratings_train)
#ratings_predict = regr.predict(features_test)
#elif args.model == 'gpr':
#	regr = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
#else:
#	raise NameError('Unknown machine learning model. Please us one of: rf, svm, linear_model, gpr')

regr = regr.fit(features_train, ratings_train)
#一定要把conpress设为true或者其他的值，没有设置会输出很多的*.npy
joblib.dump(regr, './model/my_face_rating.pkl', compress=1)

print("Generate Model Successfully!")

ratings_predict = regr.predict(features_test)
truth, = plt.plot(ratings_test, 'r')
prediction, = plt.plot(ratings_predict, 'b')
plt.legend([truth, prediction], ["Ground Truth", "Prediction"])

plt.show()