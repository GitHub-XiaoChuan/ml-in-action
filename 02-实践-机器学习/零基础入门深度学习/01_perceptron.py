from functools import reduce
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
    return 1 if x > 0 else 0

def get_training_dataset():
    input_vecs = [[1, 1], [0, 0], [1, 0], [0, 1]]
    labels = [1, 0, 0, 0]
    return input_vecs, labels

def train_and_perceptron():
    p = Perceptron(2, f)
    input_ves, labels = get_training_dataset()
    p.train(input_ves, labels, 10, 0.1)
    return p

if __name__ == '__main__':
    and_perceptron = train_and_perceptron()
    print(and_perceptron)
    print('1 and 1 = %d' % and_perceptron.predict([1, 1]))
    print('0 and 0 = %d' % and_perceptron.predict([0, 0]))
    print('1 and 0 = %d' % and_perceptron.predict([1, 0]))
    print('0 and 1 = %d' % and_perceptron.predict([0, 1]))