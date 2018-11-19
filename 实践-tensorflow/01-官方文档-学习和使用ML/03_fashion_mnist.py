from tensorflow import keras
import matplotlib.pyplot as plt

# 下载fashion mnist的数据集，图像为28*28的数组，像素值在0-255之间。
# 标签在0-9。

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 探索数据

# (60000, 28, 28)
print(train_images.shape)
# 60000
print(len(train_labels))

# (10000, 28, 28)
print(test_images.shape)
# 10000
print(len(test_labels))

# 显示一张图
plt.figure()
plt.imshow(train_images[10])
plt.colorbar()
plt.grid(False)
plt.show()

# 显示25张图片
train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure()
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.ylabel([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()