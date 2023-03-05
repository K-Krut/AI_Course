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


def get_connections_n(neurons, input_neurons):
    return input_neurons * (neurons - input_neurons) + input_neurons * input_neurons


class NeuralNetworkXOR:
    def __init__(self, data, weights=None, neurons_n=3, input_nn=2):
        self.neurons_n = neurons_n
        self.input_nn = input_nn
        if isinstance(weights, list) and len(weights) == get_connections_n(self.neurons_n, self.input_nn):
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
        return f'\n'.join([f'{i[0]} => {i[1]}' for i in self.result]) + '\n---------------\n'


data_XOR = [
    (0, 0),  # Expected output: 0
    (0, 1),  # Expected output: 1
    (1, 0),  # Expected output: 1
    (1, 1)   # Expected output: 1
]
print(NeuralNetworkXOR(data_XOR))