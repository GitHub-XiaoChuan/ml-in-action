import nltk
from nltk.corpus import movie_reviews

"""
http://www.shareditor.com/blogshow?blogId=69
5. 自然语言处理中的文本分类
"""

"""
基于贝叶斯分布判断分类
"""
my_train_set = [
        ({'feature1':u'a'},'1'),
        ({'feature1':u'a'},'2'),
        ({'feature1':u'a'},'3'),
        ({'feature1':u'a'},'3'),
        ({'feature1':u'b'},'2'),
        ({'feature1':u'b'},'2'),
        ({'feature1':u'b'},'2'),
        ({'feature1':u'b'},'2'),
        ({'feature1':u'b'},'2'),
        ({'feature1':u'b'},'2'),
        ]
classifier = nltk.NaiveBayesClassifier.train(my_train_set)
print(classifier.classify({'feature1':u'a'}))
print(classifier.classify({'feature1':u'b'}))

"""
基于词频分布的文档分类
"""
# 获得词频
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
print(all_words)
# 取词频的前2000个高频词
word_features = list(all_words.keys())[:2000]
print(word_features)

def document_features(document):
    for word in word_features:
        features = {}
        features['contains(%s)' % word] = (word in document)
        return features

print(movie_reviews.categories())

documents = [
        (['plot', 'two'],'1'),
        (['plot', 'teen'],'1'),
        (['couples', 'two'],'2'),
        (['go', 'drink'],'1'),
        (['plot', 'party'],'2'),
        (['to', 'church'],'2')
        ]

# 生成样本数据
featuresets = [(document_features(d), c) for (d,c) in documents]
print(featuresets)
# 训练
classifier = nltk.NaiveBayesClassifier.train(featuresets)
# 预测
print(classifier.classify(document_features(nltk.word_tokenize('plot two'))))
print(classifier.classify(document_features(nltk.word_tokenize('plot to'))))
print(classifier.classify(document_features(nltk.word_tokenize('couples to'))))