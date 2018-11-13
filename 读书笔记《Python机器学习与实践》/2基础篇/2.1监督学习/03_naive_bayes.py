from sklearn.datasets import fetch_20newsgroups
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# 朴素贝叶斯分类器单独考量每一维度的特征被分类的条件概率，综合判断进行分类预测
# 前提：各个维度的特征被分类的概率是相互独立的
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

news = fetch_20newsgroups(subset='all')
print(len(news.data))
print(news.data[0])

x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)

vec = CountVectorizer()
x_train = vec.fit_transform(x_train)
x_test = vec.transform(x_test)

mnb = MultinomialNB()
mnb.fit(x_train, y_train)
y_predict = mnb.predict(x_test)

print('The accuracy of Naive Bayes Classifiter is %d' % mnb.score(x_test, y_test))
print(classification_report(y_test, y_predict, target_names=news.target_names))