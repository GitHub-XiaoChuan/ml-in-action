import nltk
from nltk import load_parser

"""
http://www.shareditor.com/blogshow?blogId=71
7. 文法分析还是基于特征好啊
"""

# 文法分析在于分析词语的排序

# 文法特征的限制：句法协议、属性、约束、术语

# 特征结构
fs1 = nltk.FeatStruct(TENSE='past', NUM='sg')
print(fs1)
fs2 = nltk.FeatStruct(POS='N', AGR=fs1)
print(fs2)

# 提供的特征
cp = load_parser('/Users/xingoo/nltk_data/grammars/book_grammars/sql0.fcfg')
query = 'What cities are located in China'
for tree in cp.parse(nltk.word_tokenize(query)):
    print(tree)