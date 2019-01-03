import nltk
from pickle import dump
from pickle import load

"""
http://www.shareditor.com/blogshow?blogId=67
4. 完全自动化对语料做词性标注
"""

"""
英文词干提取器
"""
porter = nltk.PorterStemmer()
print(porter.stem('lying'))


"""
词性标注器
"""
# CC是连接词，RB是副词，IN是介词，NN是名次，JJ是形容词
text = nltk.word_tokenize("And now for something completely different")
print(nltk.pos_tag(text))

# 自定义词性
tagged_token = nltk.tag.str2tuple('fly/NN')
print(tagged_token)

# 其他语料库的标注
print(nltk.corpus.brown.tagged_words())

"""
词性自动标注
"""

# 默认标注
default_tagger = nltk.DefaultTagger('NN')
raw = '我 喜欢 你'
tokens = nltk.word_tokenize(raw)
tags = default_tagger.tag(tokens)
print(tags)

# 满足正则的标注
pattern = [('.*们$','PRO')]
tagger = nltk.RegexpTagger(pattern)
print(tagger.tag(nltk.word_tokenize('我们 喜欢 你们')))

# 查询标注
tagged_sents = [[('我','PRO'),('兔子','NN')]]
unigram_tagger = nltk.UnigramTagger(tagged_sents)
tags = unigram_tagger.tag(['我','你','兔子'])
print(tags)

# 二元标注，还考虑前面的词的标注

train_sents = []

"""
组合标注，与保存使用
"""

# 组合标注
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t2 = nltk.BigramTagger(train_sents, backoff=t1)
# 保存
output = open('t2.pkl', 'wb')
dump(t2, output, -1)
output.close()
# 加载
input = open('t2.pkl', 'rb')
tagger = load(input)
input.close()
