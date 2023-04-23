import numpy as np

"""
hidden = neurones
inp = input neurones 
out = output neurones
"""


def get_connections_n(hidden, inp, out):
    return inp * hidden + hidden * out


def f(x):
    return 1 / (1 + np.exp(-x))


def df(x):
    return x * (1 - x)


class Neuron:
    def __init__(self, weights_, name='Neuron'):  # inputs_n
        self.weights = weights_
        self.name = name
        self.delta = None
        self.neuron_fS = None

    def get_weights(self):
        return self.weights

    def set_weights(self, w):
        if isinstance(w, list):
            self.weights = w
        if w is None:
            self.weights = [1, 1, 1]

    def set_delta(self, delta):
        self.delta = delta

    def get_S(self, inputs):
        return np.dot(np.array(self.weights), inputs)

    def f(self, x):
        return 1 / (1 + np.exp(-x))

    def df(self, x):
        return x * (1 - x)  # return 1 / (1 + np.exp(-x)) * (1 - 1 / (1 + np.exp(-x)))

    def set_neuron_fS(self, fs):
        self.neuron_fS = fs

    def forward(self, inputs):
        return self.f(self.get_S(inputs))

    def __str__(self):
        return f'{self.name}\nweights: {self.weights}\nf(S): {self.neuron_fS}\ndelta: {self.delta}\n'


class NeuralNetwork:
    def __init__(self, weights_=None, iterations=1000, lmd=0.1, input_size=3, hidden_size=4, output_size=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.iterations = iterations
        self.lmd = lmd
        self.conn_n = get_connections_n(self.hidden_size, self.input_size, self.output_size)
        self.weights = weights_ if isinstance(weights_, list) and len(weights_) == self.conn_n else np.random.randn(
            self.conn_n)
        self.hidden_neurones = []
        self.output_neuron = None

    def set_iterations(self, iterations):
        self.iterations = iterations

    def set_lmd(self, lmd):
        self.lmd = lmd

    def set_weights(self, w):
        if isinstance(w, list) and len(w) == self.conn_n:
            self.weights = w

    def get_hidden_weights(self):
        return np.array_split(self.weights[:len(self.weights) - self.hidden_size], self.hidden_size)

    def get_output_weights(self):
        return self.weights[self.conn_n - self.hidden_size:]

    def initialize_hidden(self):
        weights_ = self.get_hidden_weights()
        self.hidden_neurones = [Neuron(weights_[i], f'Neuron {self.input_size + i + 1}') for i in range(len(weights_))]

    def initialize_output(self):
        weights_ = self.get_output_weights()
        if self.output_size == 1:
            self.output_neuron = Neuron(weights_, f'Neuron {self.input_size + self.output_size + self.hidden_size}')
        # else:
        #     weights_ = np.array_split(weights_, self.output_size)
        #     n = self.input_size + self.output_size + self.hidden_size
        #     self.output_neuron = [Neuron(weights_[i], f'Neuron {n + 1}') for i in range(len(weights_))]

    def get_hidden_fS(self, inputs):
        for neuron in self.hidden_neurones:
            neuron.set_neuron_fS(neuron.forward(inputs))
        return [neuron.neuron_fS for neuron in self.hidden_neurones]

    def get_out_fS(self, hidden_fs):
        self.output_neuron.set_neuron_fS(self.output_neuron.forward(hidden_fs))
        return self.output_neuron.neuron_fS

    def set_out_delta(self, expected):
        error = expected - self.output_neuron.neuron_fS
        delta = error * df(self.output_neuron.neuron_fS)
        self.output_neuron.set_delta(delta)

    def set_hidden_delta(self):
        for i in range(len(self.hidden_neurones)):
            neuron = self.hidden_neurones[i]
            neuron.set_delta(self.output_neuron.delta * self.output_neuron.weights[i] * df(neuron.neuron_fS))

    def adjust_weights_output(self, hidden_fs):
        return np.array(self.output_neuron.weights) + self.output_neuron.delta * self.lmd * np.array(hidden_fs)

    def adjust_weights(self, hidden_fs, inputs):

        hidden_weights = np.array(
            [np.array(i.weights) + i.delta * self.lmd * np.array(inputs) for i in self.hidden_neurones]
        ).flatten()

        self.weights = np.concatenate([hidden_weights, self.adjust_weights_output(hidden_fs)])

    def check(self, inputs):
        self.initialize_hidden()
        self.initialize_output()
        return self.get_out_fS(self.get_hidden_fS(np.array(inputs) / 10)) * 10

    def test(self, data_):
        return (data_[1], self.check(data_[0]))

    def train(self, data_):
        inputs = np.array(data_[0]) / 10
        expected = data_[1] / 10
        for iteration in range(self.iterations):
            self.initialize_hidden()
            self.initialize_output()
            hidden_fs = self.get_hidden_fS(inputs)
            out_fS = self.get_out_fS(hidden_fs)
            error = expected - out_fS
            delta = error * df(out_fS)
            self.output_neuron.set_delta(delta)
            self.set_hidden_delta()
            self.adjust_weights(hidden_fs, inputs)

            if iteration == 999:
                for i in self.hidden_neurones:
                    print(i)

                print(self.output_neuron)


data = [2.65, 5.60, 1.21, 5.48, 0.73, 4.08, 1.88, 5.31, 0.78, 4.36, 1.71, 5.62, 0.43, 4.21, 1.21]
# [w14, w24, w34, w15 .. w68, w78]
weights = [-0.1, 0.2, 0.1, 0.2, 0.5, -0.7, 0.1, -0.3, 0.5, 0.3, 0.6, -0.4, 0.3, 0.3, 0.2, 0.1]

nn = NeuralNetwork(weights)
nn.train(([2.65, 5.60, 1.21], 5.48))
print(nn.check([2.65, 5.60, 1.21]))
print(nn.test(([5.62, 0.43, 4.21], 1.21)))
