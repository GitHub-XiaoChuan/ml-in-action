import pandas as pd
import numpy as np
import pylab as pl
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn import feature_selection

titanic = pd.read_csv('titanic.csv')

y = titanic['survived']
x = titanic.drop(['row.names', 'name', 'survived'], axis=1)

# 缺失值补充
x['age'].fillna(x['age'].mean(), inplace=True)
x.fillna('UNKNOW', inplace=True)

# 分割数据
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.25,
                                                    random_state=33)

vec = DictVectorizer()
x_train = vec.fit_transform(x_train.to_dict(orient='record'))
x_test = vec.transform(x_test.to_dict(orient='record'))

# 输出特征的维度
print(len(vec.feature_names_))

dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train, y_train)
print(dt.score(x_test, y_test))

# 筛选20%的特征
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=20)
x_train_fs = fs.fit_transform(x_train, y_train)
x_test_fs = fs.transform(x_test)

dt.fit(x_train_fs, y_train)
print(dt.score(x_test_fs, y_test))


percentiles = range(1, 100, 2)
results = []
for i in percentiles:
    print(i)
    fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=i)
    x_train_fs = fs.fit_transform(x_train, y_train)
    # 基于交叉验证得到最终的评分
    scores = cross_val_score(dt, x_train_fs, y_train, cv=5)
    # 结果取平均值保存到results
    results = np.append(results, scores.mean())
print(results)

# 选择结果最大的那个索引值
opt = np.where(results == results.max())[0][0]

# 获得对应的百分比
print(percentiles[opt])

pl.plot(percentiles, results)
pl.xlabel('percentiles')
pl.ylabel('accuracy')
pl.show()


fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=7)
x_train_fs = fs.fit_transform(x_train, y_train)
dt.fit(x_train_fs, y_train)
x_test_fs = fs.transform(x_test)
print(dt.score(x_test_fs, y_test))