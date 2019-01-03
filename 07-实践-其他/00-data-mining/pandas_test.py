import pandas as pd
import numpy as np

# 参考文章：https://www.jianshu.com/p/7414364992e4

# 基本使用
pd1 = pd.Series(np.arange(4,10))
print(pd1)

pd2 = pd.Series([11, 12, 14], index=['北京', '上海', '深圳'])
print(pd2)

pd3 = pd.Series({'北京':11, '上海':12, '深圳':14})
print(pd3)

data_3_4 = pd.DataFrame(np.arange(10,22).reshape(3,4))
print(data_3_4)

print(data_3_4[:1])
print(data_3_4[:][0])

# 基本方法
txt = pd.read_csv("./numpy.csv")
print(txt.shape)    # 形状
print(txt.dtypes)   # 数据类型
print(txt.ndim)     # 维数
print(txt.index)    # index
print(txt.columns)  # 列
print(txt.values)   # 值

print(txt.head(2))
print(txt.tail(2))
print(txt.describe())

# 数据获取
print(txt["a"])
print(txt["a"]>6)
print(txt[txt["a"]>6])

# 应用

# 排序
print(txt.sort_values(by="d", ascending=True))
print(txt.sort_values(by="d", ascending=False))

# 统计
print(txt[txt['d']==txt['d'].max()])
print(txt[txt['d']==txt['d'].min()])
print(txt['d'].mean())

# 删除缺失值
print(txt.dropna())         # 删除空值行
print(txt.dropna(axis=1))   # 删除空值列

# 填充缺失值
txt['e'].fillna(txt['e'].mean(), inplace=True)
print(txt)

# 文本替换
txt2 = pd.read_csv("./numpy2.csv")
print(txt2)
txt2 = txt2.replace(to_replace='?', value=np.nan)
print(txt2)

# 时间戳转换
# pd.to_datetime(txt["time"], unit="s")

# 抽取年月日
# txt["year"] = pd.DatetimeIndex(txt["time"]).year

# 合并
user_info = pd.read_csv("./user_info.csv")
print(user_info)

goods_info = pd.read_csv("./goods_info.csv")
print(goods_info)

order_info = pd.read_csv("./order_info.csv")
print(order_info)

# left join
u_o = pd.merge(user_info, order_info, how='left', on=['user_id', 'user_id'])
print(u_o)

u_o_g = pd.merge(u_o, goods_info, how='left', on=['goods_name', 'goods_name'])
print(u_o_g)

# 交叉表
user_goods = pd.crosstab(u_o_g["姓名"],u_o_g["goods_name"])
print(user_goods)

