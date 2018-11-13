from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

digits = load_digits()
print(digits.data.shape) # (1797, 64)


x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=33)
print(y_train.shape) # (1347,)
print(y_test.shape) # (450,)

ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

lsvc = LinearSVC()

lsvc.fit(x_train, y_train)
y_predict = lsvc.predict(x_test)

# 对于多分类任务，把所有非当前样本的分类看做负样本
print('The Accuracy of Linear SVC is %d' % lsvc.score(x_test, y_test))
print(classification_report(y_test, y_predict, target_names=digits.target_names.astype(str)))