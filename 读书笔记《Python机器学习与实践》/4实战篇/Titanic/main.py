import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

# 1 特征补全

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(train.info())
print(test.info())

selected_features = [
    'Pclass',
    'Sex',
    'Age',
    'Embarked',
    'SibSp',
    'Parch',
    'Fare'
]

x_train = train[selected_features]
x_test = test[selected_features]

y_train = train['Survived']

# 使用频率最高的值 进行填充
print(x_train['Embarked'].value_counts())
print(x_test['Embarked'].value_counts())

x_train['Embarked'].fillna('S', inplace=True)
x_test['Embarked'].fillna('S', inplace=True)

x_train['Age'].fillna(x_train['Age'].mean(), inplace=True)
x_test['Age'].fillna(x_test['Age'].mean(), inplace=True)
x_test['Fare'].fillna(x_test['Fare'].mean(), inplace=True)

print(x_train.info())
print(x_test.info())

# 2 特征预处理
dict_vec = DictVectorizer(sparse=False)
x_train = dict_vec.fit_transform(x_train.to_dict(orient='record'))
print(dict_vec.feature_names_)
x_test = dict_vec.transform(x_test.to_dict(orient='record'))


# 3 训练
rfc = RandomForestClassifier()
print(cross_val_score(rfc, x_train, y_train, cv=5).mean())

rfc.fit(x_train, y_train)
rfc_y_predict = rfc.predict(x_test)
rfc_submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': rfc_y_predict
})
rfc_submission.to_csv('rfc_submission.csv', index=False)