from sklearn.feature_extraction import DictVectorizer

"""
文本：0 - 1 二值化
数值：维持元特征
"""

measurements = [{'city': 'Dubai', 'temprature': 33.},
                {'city': 'London', 'temprature': 12.},
                {'city': 'San Fransisico', 'temprature': 18.}]

vec = DictVectorizer()

print(vec.fit_transform(measurements).toarray())
print(vec.get_feature_names())

