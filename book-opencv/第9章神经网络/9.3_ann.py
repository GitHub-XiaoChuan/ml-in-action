import cv2
import numpy as np

# 创建ann模型
ann = cv2.ml.ANN_MLP_create()
# 设置网络结构
ann.setLayerSizes(np.array([9, 5, 9], dtype=np.uint8))
# 配置优化算法
ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
# 开始训练
ann.train(np.array([[1.2, 1.3, 1.9, 2.2, 2.3, 2.9, 3.0, 3.2, 3.3]], dtype=np.float32),
          cv2.ml.ROW_SAMPLE,
          np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0]], dtype=np.float32))
# 使用
print(ann.predict(np.array([[1.4, 1.5, 1.2, 2., 2.5, 2.8, 3., 3.1, 3.8]], dtype=np.float32)))

