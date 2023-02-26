import numpy as np


class Neuron:
    def __init__(self, weights):
        self.weights = weights

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


class NeuronAND(Neuron):
    def __init__(self, weights):
        if isinstance(weights, list):
            self.weights = weights
        if weights is None:
            self.weights = [1, 1]
        super().__init__(weights)
        self.weights = weights
        self.name = 'Neuron AND'

    def act_F(self, weighted):
        return 1 if weighted >= 1.5 else 0

    def forward(self, inputs):  # print(self.weights, inputs)   print(np.dot(self.weights, inputs))
        return self.act_F(self.get_S(inputs))

    def __str__(self):
        return f'{self.name}\nweights: {self.weights}\n'


class NeuronNOT(Neuron):
    def __init__(self, weight):
        if isinstance(weight, float):
            self.weights = weight
        if weight is None:
            self.weights = -1.5
        super().__init__(weight)
        self.weights = weight
        self.name = 'Neuron NOT'

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

    def __str__(self):
        return f'{self.name}\nweights: {self.weights}\n'


class NeuralNetworkLO:
    def __init__(self, neuron, data):
        self.neuron = neuron  # self.neuron = NeuronAND([1, 1])
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

print(NeuralNetworkLO(NeuronAND([1, 1]), data_AND))
print(NeuralNetworkLO(NeuronNOT(-1.5), data_NOT))
