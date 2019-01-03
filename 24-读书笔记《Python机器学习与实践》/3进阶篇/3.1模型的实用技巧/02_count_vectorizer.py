from sklearn.datasets import fetch_20newsgroups
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

news = fetch_20newsgroups(subset='all')

x_train, x_test, y_train, y_test = train_test_split(news.data, news.target,
                                                    test_size=0.25,
                                                    random_state=33)
# 默认的countVectorizer不会去掉停顿词
# count_vec = CountVectorizer()
count_vec = CountVectorizer(analyzer='word', stop_words='english')
x_count_train = count_vec.fit_transform(x_train)
x_count_test = count_vec.transform(x_test)

# 朴素贝叶斯
mnb_count = MultinomialNB()
mnb_count.fit(x_count_train, y_train)

print('The accuracy of classifying 20newsgroups :', mnb_count.score(x_count_test, y_test))

y_count_predict = mnb_count.predict(x_count_test)
print(classification_report(y_test, y_count_predict, target_names=news.target_names))