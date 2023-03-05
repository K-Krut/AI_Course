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
        # self.S = None

    def get_weights(self):
        return self.weights

    def set_weights(self, w):
        if isinstance(w, list):
            self.weights = w
        if w is None:
            self.weights = [1, 1, 1]

    # def set_S(self, s):
    #     self.S = s

    def get_S(self, inputs):
        return np.dot(np.array(self.weights), inputs)

    def f(self, x):
        return 1 / (1 + np.exp(-x))

    def df(self, x):
        return x * (1 - x)  # return 1 / (1 + np.exp(-x)) * (1 - 1 / (1 + np.exp(-x)))

    def forward(self, inputs):
        return self.f(self.get_S(inputs))

    def __str__(self):
        return f'{self.name}\nweights: {self.weights}\n'


"""
    def get_input_size(self):
        return self.input_size

    def get_hidden_size(self):
        return self.hidden_size

    def get_output_size(self):
        return self.output_size

    def get_iterations(self):
        return self.iterations"""


class NeuralNetwork:
    def __init__(self, weights_=None, iterations=1000, lmd=0.01, input_size=3, hidden_size=4, output_size=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.iterations = iterations
        self.lmd = lmd
        self.conn_n = get_connections_n(self.hidden_size, self.input_size, self.output_size)
        self.weights = weights_ if isinstance(weights_, list) and len(weights_) == self.conn_n else np.random.randn(self.conn_n)
        self.hidden_neurones = []  # мб добавить нейроны с названием, что бы можно было просмотреть, визуализировать энту фигню ыыыыыы
        # self.weighted = []

    def set_iterations(self, iterations):
        self.iterations = iterations

    def set_lmd(self, lmd):
        self.lmd = lmd

    def set_weights(self, w):
        if isinstance(w, list) and len(w) == self.conn_n:
            self.weights = w

    def get_hidden_weights(self):
        return np.array_split(self.weights[:len(self.weights) - self.hidden_size], self.hidden_size)

    def initialize_hidden(self):
        self.hidden_neurones = [Neuron(i, f'Neuron {self.input_size + i + 1}') for i in self.get_hidden_weights()]

    # def get_hidden_S(self, inputs=[2.65, 5.60, 1.21]):
    #     # for i in range(self.hidden_size):
    #     #     n = Neuron(hidden_weights[i], f'Neuron {self.input_size + i + 1}')
    #     #     print(n, n.get_S([2.65, 5.60, 1.21]))
    #     return [neuron.get_S(inputs) for neuron in self.hidden_neurones]

    def get_hidden_fS(self, inputs=[2.65, 5.60, 1.21]):
        return [neuron.forward(inputs) for neuron in self.hidden_neurones]

    def get_out_fS(self, inputs=[2.65, 5.60, 1.21]):
        return f(np.dot(self.get_hidden_fS(inputs), self.weights[self.conn_n - self.hidden_size:]))

    def get_adjusted_weights(self):
    def train(self):
        self.initialize_hidden()
        out_fS = self.get_out_fS()
        error = 5.48 - out_fS ##########################
        delta = error * df(out_fS)

data = [2.65, 5.60, 1.21, 5.48, 0.73, 4.08, 1.88, 5.31, 0.78, 4.36, 1.71, 5.62, 0.43, 4.21, 1.21]
# [w14, w15, w16, w17, w24 ...]
# weights = [-0.1, 0.2, 0.1, 0.3, 0.1, 0.5, -0.3, 0.6, 0.3, -0.7, 0.5, -0.4, 0.3, 0.3, 0.2, 0.1]
# [w14, w24, w34, .. ? .. ]
weights = [-0.1, 0.2, 0.1, 0.2, 0.5, -0.7, 0.1, -0.3, 0.5, 0.3, 0.6, -0.4, 0.3, 0.3, 0.2, 0.1]


def main():
    nn = NeuralNetwork(weights)



main()
print(weights[12:])

"""
Neuron 4
weights: [-0.1  0.2  0.1]
 0.9759999999999999
Neuron 5
weights: [ 0.2  0.5 -0.7]
 2.483
Neuron 6
weights: [ 0.1 -0.3  0.5]
 -0.81
Neuron 7
weights: [ 0.3  0.6 -0.4]
 3.6709999999999994
"""
# i = 0
# print(weights[i :i + 3])
