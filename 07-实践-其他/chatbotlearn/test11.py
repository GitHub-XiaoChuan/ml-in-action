import nltk
import jieba
from jieba import analyse

"""
http://www.shareditor.com/blogshow?blogId=76
11. 0字节存储海量语料资源
"""

# 关键词提取
s = '怎么才能把电脑里的垃圾文件删除'
key_words = analyse.extract_tags(s)
print(key_words)

# 利用搜索引擎