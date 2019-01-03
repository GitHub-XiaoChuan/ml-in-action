import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import brown

"""
http://www.shareditor.com/blogshow?blogId=65
3 语料与词频分布
"""

"""
常用的语料库以及API

gutenberg 约翰`古登堡，印刷《圣经》
webtext 网络文本语料
brown 文本分类语料
reuters 新闻语料
inaugural 总统就职语料
"""

print(nltk.corpus.gutenberg.fileids())

# 输出原始内容
print(nltk.corpus.gutenberg.raw('chesterton-brown.txt'))
# 输出单词列表
print(nltk.corpus.gutenberg.words('chesterton-brown.txt'))
# 输出句子列表
print(nltk.corpus.gutenberg.sents('chesterton-brown.txt'))

# 语料的通用接口
# fileids 语料中的文件
# raw 原始内容
# words 词汇
# sents 句子
# categories 分类
print(nltk.corpus.brown.categories())

# 加载自己的语料
wordlists = PlaintextCorpusReader('test3_data', '.*')
print(wordlists.fileids())

"""
统计条件频率分布
"""

# 条件频率分布，形成分类和词的对应关系
genre_word = [(genre, word)
        for genre in brown.categories()
        for word in brown.words(categories=genre)
        ]
# 创建条件频率分布
cfd = nltk.ConditionalFreqDist(genre_word)
# 指定条件和样本作图
cfd.plot(conditions=['news','adventure'], samples=[u'stock', u'sunbonnet', u'Elevated', u'narcotic', u'four', u'woods', u'railing', u'Until', u'aggression', u'marching', u'looking', u'eligible', u'electricity', u'$25-a-plate', u'consulate', u'Casey', u'all-county', u'Belgians', u'Western', u'1959-60', u'Duhagon', u'sinking', u'1,119', u'co-operation', u'Famed', u'regional', u'Charitable', u'appropriation', u'yellow', u'uncertain', u'Heights', u'bringing', u'prize', u'Loen', u'Publique', u'wooden', u'Loeb', u'963', u'specialties', u'Sands', u'succession', u'Paul', u'Phyfe'])

"""
基于词频造句子
"""

# 基于条件概率形成文本，即找到一个词概率最大的下一个词，再找下一个词，形成文本
def generate_model(cfdist, word, num=10):
    for i in range(num):
        print(word),
        word = cfdist[word].max()

text = nltk.corpus.genesis.words('english-kjv.txt')
# 形成连词
bigrams = nltk.bigrams(text)
# 形成条件频率分布
cfd = nltk.ConditionalFreqDist(bigrams)
# 以the开头造句
generate_model(cfd, 'the')

"""
其他常用的语料库资源
"""
# 词汇表
print(nltk.corpus.words.words())
# 停用词
print(nltk.corpus.stopwords.words())
# 发音
print(nltk.corpus.cmudict.dict())
# print(nltk.corpus.swadesh)
# print(nltk.WordNetLemmatizer)