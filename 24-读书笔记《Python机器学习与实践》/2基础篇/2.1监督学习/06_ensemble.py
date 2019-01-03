import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

"""
第一种 bagging 基于多个弱分类器投票，如随机森林

算法流程：
1 随机抽取n个样本，进行k轮，得到k个训练集
2 得到k个训练模型
3 对于分类，采用投票；对于回归，采用均值


第二种 boosting 基于一定次序搭建多个分类器，GTB梯度提升决策树

算法流程：
1 针对训练集中的错误样本，提升权重
2 每个迭代得到一个分类器，基于策略进行组合

区别：
1 bagging随机放回；boosting每次修改单个样本的权重
2 样本权重：bagging均匀取样；boosting错误率调整样本
3 预测函数：预测模型权重相同；预测模型根据误差改变
4 并行计算：bagging各个函数并行；Boosting按照顺序迭代


bagging + 决策树 = 随机森林
adaboost + 决策树 = 提升树
gradient boosting + 决策树 = GBDT
"""

titanic = pd.read_csv('titanic.csv')
x = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']

x['age'].fillna(x['age'].mean(), inplace=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)

vec = DictVectorizer(sparse=False)
x_train = vec.fit_transform(x_train.to_dict(orient='record'))
x_test = vec.transform(x_test.to_dict(orient='record'))

dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
dtc_y_pred = dtc.predict(x_test)

rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
rfc_y_pred = rfc.predict(x_test)

gbc = GradientBoostingClassifier()
gbc.fit(x_train, y_train)
gbc_y_pred = gbc.predict(x_test)

print('decision tree %d' % dtc.score(x_test, y_test))
print(classification_report(dtc_y_pred, y_test))

print('random forest classifiter is %d' % rfc.score(x_test, y_test))
print(classification_report(rfc_y_pred, y_test))

print('gradient tree boosting is %d' % gbc.score(x_test, y_test))
print(classification_report(gbc_y_pred, y_test))