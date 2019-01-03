import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

titanic = pd.read_csv('titanic.csv')
print(titanic.head())
print(titanic.info())

x = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']

print(x.info())

# 使用平均数作为默认值
x['age'].fillna(x['age'].mean(), inplace=True)
print(x.info())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)

vec = DictVectorizer(sparse=False)
x_train = vec.fit_transform(x_train.to_dict(orient='record'))
print(vec.feature_names_)
x_test = vec.transform(x_test.to_dict(orient='record'))
print(vec.feature_names_)

dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
y_predict = dtc.predict(x_test)

print(dtc.score(x_test, y_test))
print(classification_report(y_predict, y_test, target_names=['died', 'survived']))