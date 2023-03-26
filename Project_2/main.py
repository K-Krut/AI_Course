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
    def __init__(self, inputs, name='Neuron', weights_=None):
        self.name = name
        self.delta = None
        self.neuron_fS = None
        self.inputs = inputs
        if isinstance(weights_, list) and len(weights_) == self.inputs:
            self.weights = weights_
        else:
            self.weights = [1 for _ in range(self.inputs)]

    def get_weights(self):
        return self.weights

    def set_weights(self, w):
        if isinstance(w, list):
            self.weights = w
        if w is None:
            self.weights = [1 for _ in range(self.inputs)]

    def set_delta(self, delta):
        self.delta = delta

    def get_S(self, inputs):
        return np.dot(np.array(self.weights), inputs)

    def set_neuron_fS(self, fs):
        self.neuron_fS = fs

    def forward(self, inputs):
        return f(self.get_S(inputs))

    def __str__(self):
        return f'{self.name} ---- {self.weights}\n'
        # return f'{self.name}\nweights: {self.weights}\nf(S): {self.neuron_fS}\ndelta: {self.delta}\n'


class Layer:
    def __init__(self, size, inputs, start_n, name='Layer', w=None):
        self.size = size
        self.start_n = start_n
        self.name = name
        self.n_inputs = inputs
        self.weights = w if isinstance(w, list) and len(w) == self.size else [1 for _ in range(self.size)]
        self.neurones = []

    def initialize(self):
        self.neurones = [Neuron(self.n_inputs, f'Neuron {self.start_n + i + 1}') for i in range(self.size)]

    def get_name(self):
        return self.name

    def set_name(self, name):
        if isinstance(name, str) and len(name) > 3:
            self.name = name

    def get_weights(self):
        return self.weights

    def get_neurones_weights(self):
        return [neuron.weights for neuron in self.neurones]

    def set_weights(self, w):
        if isinstance(w, list) and len(w) == self.size:
            self.weights = w
        else:
            self.weights = [1 for _ in range(self.size)]

    def get_fS(self, inputs):
        for neuron in self.neurones:
            neuron.set_neuron_fS(neuron.forward(inputs))
        return [neuron.neuron_fS for neuron in self.neurones]

    # def get_deltas(self, errors, ):
    #     for neuron in self.neurones:
    #         neuron.set_delta(self.output_neuron.delta * self.output_neuron.weights[i] * df(neuron.neuron_fS))
    #     return [neuron.neuron_fS for neuron in self.neurones]

    def adjust_weights(self, inputs, lmd):
        self.weights = np.array(
            [np.array(neuron.weights) + neuron.delta * lmd * np.array(inputs) for neuron in self.neurones]).flatten()

    def __str__(self):
        return f'{self.name}\n' + ''.join([str(neuron) for neuron in self.neurones])


class HiddenLayer(Layer):
    def __init__(self, start_n, inputs, size=36, name='Hidden Layer', w=None):
        super().__init__(size, inputs, start_n, name, w)
        self.fS = []
        self.errors = []
        self.deltas = []

    def set_fS(self, fs):
        self.fS = fs

    def set_errors(self, errors):
        self.errors = errors

    def get_deltas(self):
        pass

    def set_deltas(self, deltas):
        self.deltas = deltas
    #     for i in range(len(self.neurones)):
    #         neuron = self.neurones[i]
    #         neuron.set_delta(self.output_neuron.delta * self.output_neuron.weights[i] * df(neuron.neuron_fS))


class OutputLayer(Layer):
    def __init__(self, size, inputs, start_n, name='Output Layer', w=None):
        super().__init__(size, inputs, start_n, name, w)
        self.fS = []
        self.errors = []
        self.deltas = []

    def set_fS(self, fs):
        self.fS = fs

    def set_errors(self, errors):
        self.errors = errors

    def get_deltas(self):
        pass

    def set_deltas(self, deltas):
        self.deltas = deltas


def get_start(arr, i):
    return 0 if i == 0 else sum(arr[0:i])


def get_inputs(input_size, arr, i):
    return input_size if i == 0 else arr[i - 1]


class NeuralNetwork:
    def __init__(self, weights_=None, iter=1000, lmd=0.1, input_size=36, hidden=[36, 36], output_size=2):
        self.input = Layer(input_size, 0, 0, 'Input Layer')
        self.hidden = [
            HiddenLayer(input_size + get_start(hidden, i), get_inputs(input_size, hidden, i), hidden[i],
                        f'Hidden Layer {i + 1}')
            for i in range(len(hidden))
        ]
        self.output = OutputLayer(output_size, hidden[-1], input_size + sum(hidden))
        self.iterations = iter
        self.lmd = lmd
        self.weights = weights_

        self.initialize()

    def initialize(self):
        self.input.initialize()
        self.output.initialize()
        for layer in self.hidden:
            layer.initialize()

    def set_iterations(self, iterations):
        self.iterations = iterations

    def set_lmd(self, lmd):
        self.lmd = lmd

    def forward(self, inputs):
        pass

    def backward(self):
        pass

    def train(self, inputs, expected):
        for layer in self.hidden:
            layer.set_fS(layer.get_fS(inputs))
        self.output.set_fS(self.output.get_fS(self.hidden[-1].fS))

        errors = np.array(expected) - np.array(self.output.fS)
        deltas = errors * df(np.array(self.output.fS))

        self.hidden.reverse()
        for i in range(len(self.hidden)):
            prev_layer = self.output if i == 0 else self.hidden[i - 1]
            layer = self.hidden[i]
            print(layer.name, prev_layer.name)
            print(deltas, prev_layer.get_neurones_weights())
            layer.set_errors(np.dot(deltas, prev_layer.get_neurones_weights()))
            layer.set_deltas(layer.errors * df(np.array(layer.fS)))

        print(errors)
        print(deltas)
        print([i.errors for i in self.hidden])
        print([i.deltas for i in self.hidden])
        self.hidden.reverse()
        print([self.hidden[i].name for i in range(len(self.hidden))])

    def __str__(self):
        return '\n----------------------\n'.join([str(self.input)] + [str(i) for i in self.hidden] + [str(self.output)])

    # def set_out_delta(self, expected):
    #     error = expected - self.output_neuron.neuron_fS
    #     delta = error * df(self.output_neuron.neuron_fS)
    #     self.output_neuron.set_delta(delta)
    #
    # def set_hidden_delta(self):
    #     for i in range(len(self.hidden_neurones)):
    #         neuron = self.hidden_neurones[i]
    #         neuron.set_delta(self.output_neuron.delta * self.output_neuron.weights[i] * df(neuron.neuron_fS))
    #
    #


arr = [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1]

nn = NeuralNetwork()
nn.train(arr, [0, 1])
