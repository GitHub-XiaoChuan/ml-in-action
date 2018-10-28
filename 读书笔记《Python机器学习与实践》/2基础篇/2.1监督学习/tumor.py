import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

# 创建特征列表
column_names = ['Sample code number',# 样本编码
                'Clump Thickness',#肿块密度
                'Uniformity of Cell Size',#细胞大小均匀性
                'Uniformity of Cell Shape',#细胞形状均匀性
                'Marginal Adhesion',#边缘粘附
                'Single Epithelial Cell Size',#单上皮细胞大小
                'Bare Nuclei',#裸核
                'Bland Chromatin',#钝染色体
                'Normal Nucleoli',#正常核仁
                'Mitoses',#有丝分裂
                'Class']#类别

# 使用pandas.read_csv读取指定数据
data = pd.read_csv('breast-cancer-wisconsin.data', names=column_names)
data = data.replace(to_replace='? ', value=np.nan)
data = data.astype(float)
data = data.dropna(how='any')
print(data.shape)

# 随机采样25%用于测试，剩下的用于训练
X_train, X_test, y_train, y_test = train_test_split(data[column_names[1:10]],
                                                    data[column_names[10]],
                                                    test_size=0.25,
                                                    random_state=33)
print(y_train.value_counts())
print(y_test.value_counts())

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

lr = LogisticRegression()
sgdc = SGDClassifier()

lr.fit(X_train, y_train)
lr_y_predict = lr.predict(X_test)
sgdc.fit(X_train, X_test)
sgdc_y_predict = sgdc.predict(X_test)

print('Accuracy of LR Classifier:',lr.score(X_test, y_test))
print(classification_report(y_test, lr_y_predict, target_names=['Benign', 'Malignant']))

print('Accuracy of SGD Classifier:', sgdc.score(X_test, y_test))
print(classification_report(y_test, sgdc_y_predict, target_names=['Benign', 'Malignant']))