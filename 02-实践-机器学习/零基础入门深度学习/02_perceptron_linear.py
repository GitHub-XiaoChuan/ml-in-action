from functools import reduce
import matplotlib.pyplot as plt

# 参考：https://www.zybuluo.com/hanbingtao/note/433855

"""
感知器可以实现

and: 0.5*x1+0.5*x2-0.8
or: 0.5*x1+0.5*x2-0.3

但是不能实现异或
"""

class Perceptron():
    def __init__(self, input_num, activator):
        self.activator = activator
        self.weights = [0.0 for _ in range(input_num)]
        self.bias = 0.0

    def __str__(self):
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

    def predict(self, input_vec):
        """
        实现 ax1+bx2+c

        :param input_vec:
        :return:
        """
        return self.activator(
            reduce(lambda a, b: a+b, map(lambda x: x[0] * x[1], zip(input_vec, self.weights)), 0.0) + self.bias
        )

    def train(self, input_vecs, labels, iteration, rate):
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)

    def _one_iteration(self, input_vecs, labels, rate):
        samples = zip(input_vecs, labels)
        for (input_vec, label) in samples:
            output = self.predict(input_vec)
            self._update_weights(input_vec, output, label, rate)

    def _update_weights(self, input_vec, output, label, rate):
        delta = label - output
        print("delta\t: %f" % delta)
        self.weights = [w + rate * delta * x for (x, w) in zip(input_vec, self.weights)]
        self.bias += rate * delta
        print(self)

def f(x):
    return x

def get_training_dataset():
    input_vecs = [[5], [3], [8], [1.4], [10.1]]
    labels = [5500, 2300, 7600, 1800, 11400]
    return input_vecs, labels

def train_linear_unit():
    lu = Perceptron(1, f)
    input_vecs, labels = get_training_dataset()
    lu.train(input_vecs, labels, 10, 0.01)
    return lu

def plot(linear_unit):
    input_vecs, labels = get_training_dataset()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter([x[0] for x in input_vecs], labels)
    weights = linear_unit.weights
    bias = linear_unit.bias
    x = range(0, 12, 1)
    y = [ weights[0]*d+bias for d in x]
    ax.plot(x, y)
    plt.show()

if __name__ == '__main__':
    linear_unit = train_linear_unit()
    print(linear_unit)
    print('work 3.4 years, salary=%.2f ' % linear_unit.predict([3.4]))
    print('work 15 years, salary=%.2f ' % linear_unit.predict([15]))
    print('work 1.5 years, salary=%.2f ' % linear_unit.predict([1.5]))
    print('work 6.3 years, salary=%.2f ' % linear_unit.predict([6.3]))
    plot(linear_unit)