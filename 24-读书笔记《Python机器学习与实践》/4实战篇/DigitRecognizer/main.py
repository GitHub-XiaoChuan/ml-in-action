import pandas as pd
from sklearn.neural_network import MLPClassifier

# 1 数据准备

train = pd.read_csv('train.csv')
print(train.shape)
test = pd.read_csv('test.csv')
print(test.shape)

y_train = train['label']
x_train = train.drop('label', 1)

x_test = test

print(x_train.shape)
print(x_test.shape)

# 2 训练模型
clf = MLPClassifier(hidden_layer_sizes=(50,),
                    activation='logistic',
                    solver='adam',
                    learning_rate='adaptive',
                    max_iter=200)
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)
print(y_predict)

# 3 格式化为提交格式
rfc_submission = pd.DataFrame({
    'ImageId': range(1, len(y_predict)+1),
    'Label': y_predict
})
rfc_submission.to_csv('clf_submission.csv', index=False)