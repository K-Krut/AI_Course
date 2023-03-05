import numpy as np


class Neuron:
    def __init__(self, weights, name='Neuron'):
        self.weights = weights
        self.name = name

    def get_weights(self):
        return self.weights

    def set_weights(self, w):
        if isinstance(w, list):
            self.weights = w
        if w is None:
            self.weights = [1, 1]

    def get_S(self, inputs):
        return np.dot(np.array(self.weights), inputs)

    def act_F(self, weighted):
        return weighted

    def forward(self, inputs):
        return self.act_F(self.get_S(inputs))

    def __str__(self):
        return f'{self.name}\nweights: {self.weights}\n'


class NeuronAND(Neuron):
    def __init__(self, weights=None, name='Neuron AND'):
        if isinstance(weights, list):
            self.weights = weights
        if weights is None:
            self.weights = [1, 1]
        super().__init__(weights)
        self.weights = weights
        self.name = name

    def act_F(self, weighted):
        return 1 if weighted >= 1.5 else 0

    def forward(self, inputs):  # print(self.weights, inputs)   print(np.dot(self.weights, inputs))
        return self.act_F(self.get_S(inputs))


class NeuronNOT(Neuron):
    def __init__(self, weight=None, name='Neuron NOT'):
        if isinstance(weight, float):
            self.weights = weight
        if weight is None:
            self.weights = -1.5
        super().__init__(weight)
        self.weights = weight
        self.name = name

    def set_weights(self, w):
        if isinstance(w, float):
            self.weights = w
        if w is None:
            self.weights = 1

    def get_S(self, inputs):
        return self.weights * inputs

    def act_F(self, weighted):
        return 1 if weighted >= -1 else 0

    def forward(self, inputs):
        return self.act_F(self.get_S(inputs))


class NeuronOR(Neuron):
    def __init__(self, weight=None, name='Neuron OR'):
        if isinstance(weight, list):
            self.weights = weight
        if weight is None:
            self.weights = [1, 1]
        super().__init__(weight)
        self.weights = weight
        self.name = name

    def act_F(self, weighted):
        return 1 if weighted >= 0.5 else 0

    def forward(self, inputs):
        return self.act_F(self.get_S(inputs))


class NeuronXOR(Neuron):
    def __init__(self, weight=None, name='Neuron XOR'):
        if isinstance(weight, list):
            self.weights = weight
        if weight is None:
            self.weights = [1, 1]
        super().__init__(weight)
        self.weights = weight
        self.name = name

    def act_F(self, weighted):
        return 1 if weighted >= 0.5 else 0

    def forward(self, inputs):
        return self.act_F(self.get_S(inputs))


class NeuralNetworkLO:
    def __init__(self, neuron, data):
        self.neuron = neuron
        self.data = data
        self.result = []

    def forward(self, inputs):
        return self.neuron.forward(inputs)

    def check_data(self):
        return [(i, self.forward(i)) for i in self.data]

    def __str__(self):
        if not self.result:
            self.result = self.check_data()
        return f'{str(self.neuron)}\n' + '\n'.join([f'{i[0]} => {i[1]}' for i in self.result]) + '\n---------------\n'


def get_connections_n(neurones, input_neurones):
    return input_neurones * (neurones - input_neurones) + input_neurones * input_neurones


class NeuralNetworkXOR:
    def __init__(self, data, weights=None, neurones_n=3, input_nn=2):
        self.neurones_n = neurones_n
        self.input_nn = input_nn
        if isinstance(weights, list) and len(weights) == get_connections_n(self.neurones_n, self.input_nn):
            self.weights = weights
        if weights is None:
            self.weights = [[1, -1], [-1, 1], [1, 1]]

        self.neurones = self.initialize()
        self.data = data
        self.result = []

    def initialize(self):  # np.array_split(np.array([1, -1, -1, 1, 1, 1]), self.neurones_n)
        return [NeuronXOR(i) for i in self.weights]

    def forward(self, inputs):
        return self.neurones[self.input_nn:][0].forward([i.forward(inputs) for i in self.neurones[:self.input_nn]])

    def check_data(self):
        return [(i, self.forward(i)) for i in self.data]

    def __str__(self):
        if not self.result:
            self.result = self.check_data()
        return ''.join([str(n) for n in self.neurones]) + '\n' + '\n'.join(
            [f'{i[0]} => {i[1]}' for i in self.result]) + '\n---------------\n'


data_AND = [
    (0, 0),  # Expected output: 0
    (0, 1),  # Expected output: 0
    (1, 0),  # Expected output: 0
    (1, 1)  # Expected output: 1
]
data_NOT = [
    0,  # Expected output: 1
    1,  # Expected output: 0
]
data_OR = [
    (0, 0),  # Expected output: 0
    (0, 1),  # Expected output: 1
    (1, 0),  # Expected output: 1
    (1, 1)  # Expected output: 1
]
data_XOR = [
    (0, 0),  # Expected output: 0
    (0, 1),  # Expected output: 1
    (1, 0),  # Expected output: 1
    (1, 1)  # Expected output: 1
]

print(NeuralNetworkLO(NeuronAND([1, 1]), data_AND))
print(NeuralNetworkLO(NeuronNOT(-1.5), data_NOT))
print(NeuralNetworkLO(NeuronOR([1, 1]), data_OR))
print(NeuralNetworkXOR(data_XOR))
