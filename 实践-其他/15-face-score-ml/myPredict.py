from sklearn.externals import joblib
import numpy as np
from sklearn import decomposition

# use your own path
clf = joblib.load('model/my_face_rating.pkl')
features = np.loadtxt('data/features_ALL.txt', delimiter=',')
my_features = np.loadtxt('data/my_features.txt', delimiter=',')
pca = decomposition.PCA(n_components=20)
pca.fit(features)

predictions = np.zeros([6, 1])

for i in range(0, 6):
    features_test = features[i, :]
    features_test = pca.transform(features_test.reshape(1,-1))
    # regr = linear_model.LinearRegression()
    # regr.fit(features_train, ratings_train)
    predictions[i] = clf.predict(features_test)
# predictions = clf.predict(features)
print(predictions)
