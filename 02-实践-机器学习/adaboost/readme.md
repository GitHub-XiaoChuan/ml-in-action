有几种boost算法：

- discrete adaboost
- real adaboost
- gentile adaboost
- logitboost

后面那俩不懂。前面两种就是弱分类分类出的值，一个是1或者-1，一个是概率。

https://blog.csdn.net/stf1065716904/article/details/77841412

算法的差异：

- 整体效果而言，效果由好到差的顺序为Logit Boost，Gentle AdaBoost， Real AdaBoost， Discrete AdaBoost
- 若弱分类器采用树桩模型（也就是只要2个叶子节点的决策树），Discrete AdaBoost的结果比其他3种算法结果差了很多，大概是由于系统偏差过大导致的泛化误差较大
- 若弱分类器采用多层的决策树（4或8个叶子节点），Discrete AdaBoost的结果能有较大提升，而其他3种算法则差异不大。
