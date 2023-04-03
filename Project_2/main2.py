import numpy as np


def f(x):
    return 1 / (1 + np.exp(-x))


def df(x):
    return x * (1 - x)

#  # relu
# def f(x):
#     return np.maximum([0.0], x)
#
#
# def df(x):
#     return np.where(x > 0, 1, 0)
#
#  # tanh
# def f(x):
#     return np.tanh(x)
#
#
# def df(x):
#     return 1 - np.tanh(x) ** 2


class NeuralNetwork:
    hidden_sum = []

    def __init__(self, inputs=36, hidden=[36], output=2, lmd=0.1, iter=1000):
        self.hidden = hidden
        if len(hidden) > 1:
            self.w_hidden = np.random.randn(inputs, hidden[0])
            self.w_out = [np.random.randn(hidden[i], hidden[i + 1]) for i in range(len(hidden) - 1)]
            self.w_out.append(np.random.randn(hidden[len(hidden) - 1], output))
        else:
            self.w_hidden = np.random.randn(inputs, hidden[0])
            self.w_out = np.random.randn(hidden[0], output)
        self.out = None
        self.lmd = lmd
        self.iter = iter

    def forward_propagation(self, inputs):
        if len(self.hidden) > 1:
            self.hidden_sum = []
            self.hidden_sum.append(f(np.dot(inputs, self.w_hidden)))
            for w in self.w_out:
                self.hidden_sum.append(f(np.dot(self.hidden_sum[-1], w)))
            self.out = self.hidden_sum[-1]
        else:
            self.hidden_sum = f(np.dot(inputs, self.w_hidden))
            self.out = f(np.dot(self.hidden_sum, self.w_out))

    def back_propagation(self, inputs, delta):
        if len(self.hidden) > 1:
            d_temp = np.dot(delta, self.w_out[-1].T)
            wd_after = np.outer(self.hidden_sum[-2], delta) * self.lmd
            wd_hidden = []
            d_hidden = []
            for i in range(1, len(self.hidden) + 1):
                d_hidden = d_temp * f(self.hidden_sum[-i - 1])
                if i < len(self.hidden):
                    d_temp = np.dot(d_hidden, self.w_out[-i - 1].T)
                    wd_hidden[:0] = np.array([np.outer(self.hidden_sum[-i - 2], d_hidden) * self.lmd])
            wd_hidden.append(wd_after)
            return np.outer(inputs, d_hidden) * self.lmd, wd_hidden
        else:
            d_hidden = np.dot(delta, self.w_out.T) * f(self.hidden_sum)
            wd_after = np.outer(self.hidden_sum, delta) * self.lmd
            wd_before = np.outer(inputs, d_hidden) * self.lmd
            return wd_before, wd_after

    def train(self, inputs, expected):
        for _ in range(self.iter):
            self.forward_propagation(inputs)
            error = expected - self.out
            delta = error * df(self.out)

            w_hidden, w_out = self.back_propagation(inputs, delta)

            self.w_hidden += w_hidden
            for weight, weight_delta in zip(self.w_out, w_out):
                weight += weight_delta

        print(f"Expected: {expected}\nActual: {self.out}\n")


arr = [1, 1, 0, 0, 0, 0,
       1, 1, 0, 0, 0, 0,
       1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1,
       0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0]

nn = NeuralNetwork()
nn.train(arr, [0, 1])
