import nltk
import jieba
import jieba.posseg as psg
from collections import Counter
from jieba import analyse

"""
http://www.shareditor.com/blogshow?blogId=74
10. 半个小时搞定词性标注与关键词提取
"""

"""
问句的解析过程

主要的技术有：分词、词性标注、命名实体识别、关键词提取、句法分析、查询问句分类

使用的工具：
1 哈工大LTP
2 jieba
"""

"""
jieba的使用：分词
"""

s = '我想和女朋友一起去北京故宫博物院参观和闲逛。'
# 精确模式
print(','.join(jieba.cut(s)))
# 全模式
print(','.join(jieba.cut(s, cut_all=True)))
# 搜索引擎模式
print(','.join(jieba.cut_for_search(s)))

"""
jieba的使用：词性标注
"""
# 词性标注
print([(x.word, x.flag) for x in psg.cut(s)])

# 过滤名词
print([(x.word, x.flag) for x in psg.cut(s) if x.flag.startswith('n')])

# 获取频率top n的词
c = Counter(nltk.corpus.gutenberg.words('chesterton-brown.txt')).most_common(20)
print(c)

# 使用自定义词典
txt = u'欧阳建国是创新办主任也是欢聚时代公司云计算方面的专家'
print(','.join(jieba.cut(txt)))

jieba.load_userdict('user_dict.txt')
print(','.join(jieba.cut(txt)))

"""
jieba词性：
形容词(1个一类，4个二类)
a 形容词
ad 副形词
an 名形词
ag 形容词性语素
al 形容词性惯用语

区别词(1个一类，2个二类)
b 区别词
bl 区别词性惯用语

连词(1个一类，1个二类)
c 连词
cc 并列连词

副词(1个一类)
d 副词

叹词(1个一类)
e 叹词

方位词(1个一类)
f 方位词

前缀(1个一类)
h 前缀

后缀(1个一类)
k 后缀

数词(1个一类，1个二类)
m 数词
mq 数量词

名词 (1个一类，7个二类，5个三类)
名词分为以下子类：
n 名词
nr 人名
nr1 汉语姓氏
nr2 汉语名字
nrj 日语人名
nrf 音译人名
ns 地名
nsf 音译地名
nt 机构团体名
nz 其它专名
nl 名词性惯用语
ng 名词性语素

拟声词(1个一类)
o 拟声词

介词(1个一类，2个二类)
p 介词
pba 介词“把”
pbei 介词“被”

量词(1个一类，2个二类)
q 量词
qv 动量词
qt 时量词

代词(1个一类，4个二类，6个三类)
r 代词
rr 人称代词
rz 指示代词
rzt 时间指示代词
rzs 处所指示代词
rzv 谓词性指示代词
ry 疑问代词
ryt 时间疑问代词
rys 处所疑问代词
ryv 谓词性疑问代词
rg 代词性语素

处所词(1个一类)
s 处所词

时间词(1个一类，1个二类)
t 时间词
tg 时间词性语素

助词(1个一类，15个二类)
u 助词
uzhe 着
ule 了 喽
uguo 过
ude1 的 底
ude2 地
ude3 得
usuo 所
udeng 等 等等 云云
uyy 一样 一般 似的 般
udh 的话
uls 来讲 来说 而言 说来
uzhi 之
ulian 连 （“连小学生都会”）

动词(1个一类，9个二类)
v 动词
vd 副动词
vn 名动词
vshi 动词“是”
vyou 动词“有”
vf 趋向动词
vx 形式动词
vi 不及物动词（内动词）
vl 动词性惯用语
vg 动词性语素

标点符号(1个一类，16个二类)
w 标点符号
wkz 左括号，全角：（ 〔 ［ ｛ 《 【 〖 〈 半角：( [ { <
wky 右括号，全角：） 〕 ］ ｝ 》 】 〗 〉 半角： ) ] { >
wyz 左引号，全角：“ ‘ 『
wyy 右引号，全角：” ’ 』
wj 句号，全角：。
ww 问号，全角：？ 半角：?
wt 叹号，全角：！ 半角：!
wd 逗号，全角：， 半角：,
wf 分号，全角：； 半角： ;
wn 顿号，全角：、
wm 冒号，全角：： 半角： :
ws 省略号，全角：…… …
wp 破折号，全角：—— －－ ——－ 半角：--- ----
wb 百分号千分号，全角：％ ‰ 半角：%
wh 单位符号，全角：￥ ＄ ￡ ° ℃ 半角：$

字符串(1个一类，2个二类)
x 字符串
xx 非语素字
xu 网址URL

语气词(1个一类)
y 语气词(delete yg)

状态词(1个一类)
z 状态词
"""


"""
使用
"""
# 分词与词性标注
s = '聊天机器人到底该怎么做呢？'
print([(x.word, x.flag) for x in psg.cut(s)])

# 关键词提取
keywords = analyse.extract_tags(s)
print(keywords)

text = "线程是程序执行时的最小单位，它是进程的一个执行流，\
        是CPU调度和分派的基本单位，一个进程可以由很多个线程组成，\
        线程间共享进程的所有资源，每个线程有自己的堆栈和局部变量。\
        线程由CPU独立调度执行，在多CPU环境下就允许多个线程同时运行。\
        同样多线程也可以实现并发操作，每个请求分配一个线程来处理。"
print(analyse.extract_tags(text))