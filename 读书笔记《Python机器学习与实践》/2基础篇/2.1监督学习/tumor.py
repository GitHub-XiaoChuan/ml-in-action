import pandas as pd
import numpy as np

# 创建特征列表
column_names = ['Sample code number',# 样本编码
                'Clump Thickness',#肿块密度
                'Uniformity of Cell Size',#细胞大小均匀性
                'Uniformity of Cell Shape',#细胞形状均匀性
                'Marginal Adhesion',#边缘粘附
                'Single Epithelial Cell Size',#单上皮细胞大小
                'Bare Nuclei',#裸核
                'Bland Chromatin',#钝染色体
                'Normal Nucleoli',#正常核仁
                'Mitoses',#有丝分裂
                'Class']#类别

# 使用pandas.read_csv读取指定数据
data = pd.read_csv('breast-cancer-wisconsin.data', names=column_names)

data = data.replace(to_replace='? ', value=np.nan)
data = data.dropna(how='any')
print(data.shape)