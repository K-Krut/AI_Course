import numpy as np


def f(x):
    return 1 / (1 + np.exp(-x))


def df(x):
    return x * (1 - x)


def relu(x):
    return np.maximum([0.0], x)


def d_relu(x):
    return np.where(x > 0, 1, 0)


def activate_tanh(x):
    return np.tanh(x)


def deactivate_tanh(x):
    return 1 - np.tanh(x) ** 2


class Network:
    activate_hidden = []

    def __init__(self, inputs=36, hidden=[36], output=2, lmd=0.1):

        self.w_hidden = np.random.randn(inputs, hidden[0])
        self.w_out = np.random.randn(hidden[0], output)
        if len(hidden) > 1:
            self.w_hidden = np.random.randn(inputs, hidden[0])
            self.w_out = [np.random.randn(hidden[i], hidden[i + 1]) for i in range(len(hidden) - 1)]
            self.w_out.append(np.random.randn(hidden[len(hidden) - 1], output))

        self.out = None
        self.lmd = lmd

    def forward_propagation(self, inputs):
        self.activate_hidden = f(np.dot(inputs, self.w_hidden))
        self.out = f(np.dot(self.activate_hidden, self.w_out))


    def train(self, inputs, expected):
        for epoch in range(100):
            self.forward_propagation(inputs)
            error = expected - self.out
            delta = error * df(self.out)

            deltas_hidden = np.dot(delta, self.w_out.T) * df(self.activate_hidden)
            w_hidden = np.outer(inputs, deltas_hidden) * self.lmd
            w_out = np.outer(self.activate_hidden, delta) * self.lmd

            self.w_hidden += w_hidden
            for weight, weight_delta in zip(self.w_out, w_out):
                weight += weight_delta

        print(f"Expected value - {expected}\tActual value - {self.out}\n")


arr = [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1]
net = Network()
net.train(arr, [0, 1])
