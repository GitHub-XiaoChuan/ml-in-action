from sklearn.datasets import fetch_20newsgroups
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
import numpy as np

news = fetch_20newsgroups(subset='all')

x_train, x_test, y_train, y_test = train_test_split(news.data[:3000],
                                                    news.target[:3000],
                                                    test_size=0.25,
                                                    random_state=33)

# 使用pipeline简化搭建流程
clf = Pipeline([
    ('vect', TfidfVectorizer(stop_words='english', analyzer='word')),
    ('svc', SVC())
])

# 设置2个超参数分别个数是4 3
# 一共组合出12中模型
parameters = {
    'svc__gamma': np.logspace(-2, 1, 4),
    'svc__C': np.logspace(-1, 1, 3)
}

# 使用网格搜索
# refit 为 True 代表最终选择一个效果最好的
# 3折交叉验证
# gs = GridSearchCV(clf, parameters, verbose=2, refit=True, cv=3)
# 并行搜索
# n_jobs=-1代表使用全部资源
gs = GridSearchCV(clf, parameters, verbose=2, refit=True, cv=3, n_jobs=-1)

# 执行网格搜索
gs.fit(x_train, y_train)

# 输出最佳的参数
print(gs.best_params_)
print(gs.best_score_)
print(gs.score(x_test, y_test))