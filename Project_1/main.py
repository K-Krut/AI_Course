import numpy as np


class Neuron:
    def __init__(self, weights):
        self.weights = weights

    def get_weights(self):
        return self.weights

    def set_weights(self, w):
        self.weights = w

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

    def set_weights(self, w):
        if isinstance(w, list):
            self.weights = w
        if w is None:
            self.weights = [1, 1]

    def act_F(self, weighted):
        return 1 if weighted >= 1.5 else 0

    def forward(self, inputs):
        # print(self.weights, inputs)   print(np.dot(self.weights, inputs))
        return self.act_F(self.get_S(inputs))



class NeuralNetworkLO:
    def __init__(self, neuron, data, weights=None):
        self.neuron = neuron  # self.neuron = NeuronAND([1, 1])
        self.neuron.set_weights(weights)
        self.data = data

    def forward(self, inputs):
        return self.neuron.forward(inputs)
    
    def check_data(self):
        for i in self.data:
            print(self.forward(i))


neuron_types = {
    'AND': NeuronAND(None)
}


data_AND = [
    (0, 0),  # Expected output: 0
    (0, 1),  # Expected output: 0
    (1, 0),  # Expected output: 0
    (1, 1)   # Expected output: 1
]

nn = NeuralNetworkLO(NeuronAND([1, 1]), data_AND)




# Test the neural network with inputs (0, 0), (0, 1), (1, 0), and (1, 1)
print(nn.forward(np.array([0, 0])))  # Expected output: 0
print(nn.forward(np.array([0, 1])))  # Expected output: 0
print(nn.forward(np.array([1, 0])))  # Expected output: 0
print(nn.forward(np.array([1, 1])))  # Expected output: 1
