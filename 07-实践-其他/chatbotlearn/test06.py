import nltk
from nltk.corpus import conll2000

"""
http://www.shareditor.com/blogshow?blogId=70
6. 教你怎么从一句话里提取出十句话的信息
"""

# 分块标记与存储

# 关系抽取
print(conll2000.chunked_sents('train.txt')[99])