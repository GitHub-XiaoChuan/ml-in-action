from nltk.book import *
"""
http://www.shareditor.com/blogshow?blogId=64
初识NLTK库
"""

# 加载书籍，输出第一本书名
print(text1)
# 搜索文本
text1.concordance("former")
# 搜索相关词
text1.similar("ship")
# 查看词在文章的位置
text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])

# 返回总字数
print(len(text1))
# 返回文本所有词的集合
print(set(text1))
# 返回文本的词的个数
print(len(set(text4)))
# 返回is这个词出现的次数
print(text4.count("is"))
# 统计词频，按照从大到小的列表
fdist1 = FreqDist(text1)
fdist1.plot(50, cumulative=True)
# 返回只出现一次的词
fdist1.hapaxes()
# 频繁关联的两个词
text4.collocations()

