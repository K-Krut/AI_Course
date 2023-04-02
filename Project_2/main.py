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
        return f'{self.name} ---- f(S): {self.neuron_fS} ---- delta: {self.delta}\n'
        # return f'{self.name}\nweights: {self.weights}\nf(S): {self.neuron_fS}\ndelta: {self.delta}\n'


class Layer:
    def __init__(self, size, inputs, start_n, name='Layer', w=None):
        self.size = size
        self.start_n = start_n
        self.name = name
        self.n_inputs = inputs
        self.weights = np.array(w if isinstance(w, list) and len(w) == self.size else [1 for _ in range(self.size)])
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

    def adjust_weights(self, fs, lmd):
        for i, neuron in enumerate(self.neurones):
            neuron.weights = [w * lmd * fs[j] for j, w in enumerate(neuron.weights)]
        self.weights = [neuron.weights for neuron in self.neurones]
        # self.weights = [w for weights in [neuron.weights for neuron in self.neurones] for w in weights]

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
        for i, neuron in enumerate(self.neurones):
            neuron.set_delta(self.deltas[i])


class OutputLayer(Layer):
    def __init__(self, size, inputs, start_n, name='Output Layer', w=None):
        super().__init__(size, inputs, start_n, name, w)
        self.fS = []
        self.errors = []
        self.deltas = []
        self.weights = np.array([1 for _ in range(self.n_inputs * self.size)])

    def set_fS(self, fs):
        self.fS = fs

    def set_errors(self, errors):
        self.errors = errors

    def get_deltas(self):
        pass

    def set_deltas(self, deltas):
        self.deltas = deltas
        for i, neuron in enumerate(self.neurones):
            neuron.set_delta(self.deltas[i])


def get_start(arr, i):
    return 0 if i == 0 else sum(arr[0:i])


def get_inputs(input_size, arr, i):
    return input_size if i == 0 else arr[i - 1]


class NeuralNetwork:
    def __init__(self, weights_=None, iter=1000, lmd=0.1, input_size=36, hidden=[36, 36], output_size=2):
        self.input = Layer(input_size, 0, 0, 'Input Layer')
        self.hidden = [
            HiddenLayer(input_size + get_start(hidden, i), get_inputs(input_size, hidden, i), hidden[i],
                        f'Hidden Layer {i + 1}') for i in range(len(hidden))
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
        for i, layer in enumerate(self.hidden):
            layer.set_fS(layer.get_fS(inputs if i == 0 else self.hidden[i - 1].fS))
        self.output.set_fS(self.output.get_fS(self.hidden[-1].fS))

    def check(self, inputs):
        inputs = np.array(inputs)
        self.forward(inputs)
        return self.output.fS

    def train(self, inputs, expected):
        inputs = np.array(inputs)
        expected = np.array(expected)

        for _ in range(self.iterations):
            self.forward(inputs)

            errors = np.array(expected) - np.array(self.output.fS)
            deltas = errors * df(np.array(self.output.fS))
            self.output.set_errors(errors)
            self.output.set_deltas(deltas)

            ##############################################################################

            self.hidden.reverse()

            for i, layer in enumerate(self.hidden):
                prev_layer = self.output if i == 0 else self.hidden[i - 1]
                delta = deltas if i == 0 else self.hidden[i - 1].deltas
                layer.set_errors(np.dot(delta, prev_layer.get_neurones_weights()))
                layer.set_deltas(layer.errors * df(np.array(layer.fS)))

            self.output.adjust_weights(self.hidden[0].fS, self.lmd)  # self.output.adjust_weights(self.hidden[-1].fS, self.lmd)

            for i, layer in enumerate(self.hidden):
                layer.adjust_weights(self.hidden[i - 1].fS, self.lmd)

            self.hidden.reverse()


        print('\n----------------------\n'.join([str(i) for i in self.hidden] + [str(self.output)]))

    def __str__(self):
        return '\n----------------------\n'.join([str(self.input)] + [str(i) for i in self.hidden] + [str(self.output)])



arr = [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1]

nn = NeuralNetwork()
nn.train(arr, [0, 1])
print(nn.check(arr))
