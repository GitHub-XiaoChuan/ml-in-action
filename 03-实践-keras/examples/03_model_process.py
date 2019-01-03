from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

################################### 模型搭建 ###################################

model = Sequential()

# 常用层（Core）
# 卷积层（Convolutional）
# 池化层（Pooling）
# 局部连接层
# 递归层（Recurrent）
# 嵌入层（ Embedding）
# 高级激活层
# 规范层
# 噪声层
# 包装层
# 当然也可以编写自己的层

# 全连接层100个节点,activation该层的激活器

# activation 激活函数
# use_bias 使用偏置
# kernel_initializer 权重初始化方法
# bias_initializer 偏置初始化
# kernel_regularizer 权重正则化
# bias_regularizer 偏置正则化
# units 输出的维度
model.add(Dense(units=64, activation='relu', input_dim=100))
# 全连接层10个节点，trainable=False表示此层的权重不进行更新
model.add(Dense(units=10,trainable=False))
# 激活函数softmax
model.add(Activation("softmax"))
#删除最后一层模型
model.pop()


################################### 模型编译 ###################################

# 完成模型的搭建后，我们需要使用.compile()方法来编译模型：

# loss损失函数，交叉熵
# optimizer优化器
# sgd随机梯度下降法进行网络训练
# metrics评估模型
# accuracy准确率作为评判结果

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 优化器

# lr学习速率
# momentum表示动量项
# decay是学习速率的衰减系数(每个epoch衰减一次)，
# nesterov的值是False或者True，表示使不使用Nesterov momentum
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# 目标函数（损失函数）：
# mean_squared_error，
# mean_absolute_error，
# squared_hinge，
# hinge，
# binary_crossentropy对数损失函数，
# categorical_crossentropy多分类的对数损失函数
model.compile(loss='categorical_crossentropy', optimizer=sgd)

################################### 模型训练 ###################################

# epochs 迭代次数
# batch_size 每次迭代使用的样本数
# shuffl 训练集是否洗乱

x_train = []
y_train = []
model.fit(x_train, y_train, epochs=5, batch_size=32, shuffle=True)

# 设置早停
from keras.callbacks import EarlyStopping

# 提前停止触发的条件
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

# validation_split 交叉验证的分割比
# callbacks 每次训练后的回调函数
hist = model.fit(x_train, y_train, validation_split=0.2, callbacks=[early_stopping])
print(hist.history)

################################### 模型评估 ###################################

x_test = []
y_test = []
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

################################### 模型预测 ###################################

classes = model.predict(x_test, batch_size=128)

from keras import backend as K
get_3rd_layer_output = K.function([model.layers[0].input], [model.layers[3].output])
layer_output = get_3rd_layer_output([x_test])[0]

################################### 存储加载 ###################################

# 保存模型成文件HDF5文件。该文件将包含：
# 1、模型的结构，以便重构该模型，
# 2、模型的权重，
# 3、训练配置（损失函数，优化器等）
# 4、优化器的状态，以便于从上次训练中断的地方开始
model.save('dnn.h5')

from keras.models import load_model
model = load_model('dnn.h5')

# 只保存结构，不保存权重和配置

# json格式
json_string = model.to_json()
from keras.models import model_from_json
model = model_from_json(json_string)

# yaml格式
yaml_string = model.to_yaml()
from keras.models import model_from_yaml
model = model_from_yaml(yaml_string)

model = Sequential()
model.save_weights('weights.h5')
model.load_weights('weights.h5')