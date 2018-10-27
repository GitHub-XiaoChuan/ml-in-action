__author__ = 'xingoo'
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

"""
特征降维
"""
def value_preprocess(arr, index1, index2):
    """
    通过合并两列生成统一的onehot

    :param arr:     原始数组
    :param index1:  第一列的索引
    :param index2:  第二列的索引
    :return: lable_encoder, onehot_encoder
    """

    # 合并两列并去重
    distinct_arr = np.unique(np.hstack((arr[:, index1], arr[:, index2])))
    reshape_arr = np.reshape(distinct_arr, [len(distinct_arr),-1])

    # 训练label
    lable_encoder = LabelEncoder()
    lable_encoder.fit(reshape_arr)
    labels = lable_encoder.transform(reshape_arr)

    # 训练one hot
    onehot_encoder = OneHotEncoder()
    onehot_encoder.fit(np.reshape(labels, [len(labels), -1]))

    return lable_encoder, onehot_encoder

def read_list(path):
    """
    读取文件内容，形成numpy 数组

    :param path: 文件路径
    :return: numpy array
    """
    list = []

    # 读取文件
    with open(path,'r') as load_f:
        for line in load_f.readlines():
            # json.dumps 是把json编程字符串
            # json.loads 是把字符串编程json
            line_json = json.loads(line.strip())
            # print(line_json)
            list.append([int(line_json['b1']),int(line_json['b2']),int(line_json['c1']),int(line_json['c2']),int(line_json['member_id']),int(line_json['sex'])])

    # 初始化numpy
    return np.array(list)

def load_data(path):
    """
    读取数据内容，并实现特征的处理和组合

    :param path:
    :return:
    """
    origin = read_list(path)

    # 得到品牌特征训练器
    brand_label_encoder, brand_onehot_encoder = value_preprocess(origin, 0, 1)

    brand_labels = brand_label_encoder.transform(np.reshape(origin[:,0], [len(origin),-1]))
    brand_onehots = brand_onehot_encoder.transform(np.reshape(brand_labels, [len(brand_labels), -1]))

    # 得到分类特征的训练器
    category_label_encoder, category_onehot_encoder = value_preprocess(origin, 2, 3)

    category_labels = category_label_encoder.transform(np.reshape(origin[:,2], [len(origin),-1]))
    category_onehots = category_onehot_encoder.transform(np.reshape(category_labels, [len(category_labels), -1]))

    data = np.hstack((brand_onehots.toarray(), category_onehots.toarray(), np.reshape(origin[:,5],[len(origin),-1])))

    # 平衡样本数据
    df = pd.DataFrame(data)
    sample = df[df[5145]==2].sample(frac=8064/55031, replace=False)
    union = pd.concat([df[df[5145]==1], sample])

    # print(union[:][np.arange(0,5145)])
    # print(union[:][5145])

    return union[:][np.arange(0,5145)], union[:][5145]

def logistic_regression(x_train, y_train):
    """
    模型训练与预测

    :param x_train:
    :param y_train:
    :return:
    """

    # 选择算法
    logistic = LogisticRegression()
    pca = PCA()
    pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

    pca.fit(x_train)

    n_components = [20, 40, 64]
    Cs = np.logspace(-4, 4, 3)

    # 训练模型
    estimator = GridSearchCV(pipe,
                             dict(pca__n_components=n_components,
                                  logistic__C=Cs))
    estimator.fit(x_train, y_train)

    print(estimator.best_score_)


if __name__=='__main__':
    x, y = load_data("data.tg")
    logistic_regression(x, y)