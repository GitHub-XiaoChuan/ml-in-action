from sklearn.datasets import fetch_20newsgroups
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

news = fetch_20newsgroups(subset='all')

x_train, x_test, y_train, y_test = train_test_split(news.data, news.target,
                                                    test_size=0.25,
                                                    random_state=33)

# tfidf_vec = TfidfVectorizer()
tfidf_vec = TfidfVectorizer(analyzer='word', stop_words='english')

x_tfidf_train = tfidf_vec.fit_transform(x_train)
x_tfidf_test = tfidf_vec.transform(x_test)

# 朴素贝叶斯
mnb_count = MultinomialNB()
mnb_count.fit(x_tfidf_train, y_train)

print('The accuracy of classifying 20newsgroups :', mnb_count.score(x_tfidf_test, y_test))

y_count_predict = mnb_count.predict(x_tfidf_test)
print(classification_report(y_test, y_count_predict, target_names=news.target_names))