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
        return f'{self.name}\nweights: {self.weights}\nf(S): {self.neuron_fS}\ndelta: {self.delta}'


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

    def set_iterations(self, iterations):
        self.iterations = iterations

    def set_lmd(self, lmd):
        self.lmd = lmd

    def set_weights(self, w):
        if isinstance(w, list) and len(w) == self.conn_n:
            self.weights = w

    def get_hidden_weights(self):
        return np.array_split(self.weights[:len(self.weights) - self.hidden_size], 4)

    def initialize_hidden(self):
        weights_ = self.get_hidden_weights()
        self.hidden_neurones = [Neuron(weights_[i], f'Neuron {self.input_size + i + 1}') for i in range(len(weights_))]

    def get_hidden_fS(self, inputs):
        for neuron in self.hidden_neurones:
            neuron.set_neuron_fS(neuron.forward(inputs))
        return [neuron.neuron_fS for neuron in self.hidden_neurones]

    def get_out_fS(self, hidden_fS):
        return f(np.dot(hidden_fS, self.weights[self.conn_n - self.hidden_size:]))

    def train(self):
        for iteration in range(1000):
            self.initialize_hidden()
            inputs = [0.265, 0.560, 0.121]
            output_weights = self.weights[self.conn_n - self.hidden_size:]
            hidden_fS = self.get_hidden_fS(inputs)
            out_fS = self.get_out_fS(hidden_fS)
            error = 0.548 - out_fS
            delta = error * df(out_fS)

            # get adjusted weights for hidden
            for i in range(len(self.hidden_neurones)):
                neuron = self.hidden_neurones[i]
                neuron.set_delta(delta * output_weights[i] * df(neuron.neuron_fS))

            res = []
            for i in self.hidden_neurones:
                res = np.concatenate([res, np.array(i.weights) + i.delta * self.lmd * np.array(inputs)])

            # get adjusted weights for output
            result_w_out = np.array(output_weights) + delta * self.lmd * np.array(hidden_fS)

            self.weights = np.concatenate([res, result_w_out])
            # print(self.weights)

        self.initialize_hidden()
        out_fS = self.get_out_fS(self.get_hidden_fS([0.265, 0.560, 0.121])) * 10
        for i in self.hidden_neurones:
            print(i)
        print(out_fS)

    def forward(self, inputs, weights_):
        return f(np.dot(inputs, weights_))

    def train2(self):
        inputs = np.array([[2.65, 5.60, 1.21]])
        output_data = np.array([[5.48]])

        weights1 = np.array([[-0.1, 0.2, 0.1, 0.2], [0.5, -0.7, 0.1, -0.3], [0.5, 0.3, 0.6, -0.4]])
        weights2 = np.array([[0.3], [0.3], [0.2], [0.1]])
        for iteration in range(1000):
            hidden_fS = self.forward(inputs, weights1)  # F(S4), F(S5)...
            out_fS = self.forward(hidden_fS, weights2) * 10
            error = output_data - out_fS
            delta = error * df(out_fS / 10)
            delta_w_out = delta * np.multiply(weights2.T, df(hidden_fS))

            weights2 += np.dot(hidden_fS.T, 0.1 * delta)
            weights1 += np.dot(inputs.T, 0.1 * delta_w_out)

        the_shape = output_data.shape
        checked_studying = f(np.dot(f(np.dot(inputs, weights1)), weights2)) * 10

        print("Check studying:")
        print("Target output 		Calculated result")
        for temp_ind in range(the_shape[0] * the_shape[1]):
            # print('                 temp_ind' , temp_ind)
            print(f"	{output_data[temp_ind][0]} 			 {checked_studying[temp_ind][0]}")


data = [2.65, 5.60, 1.21, 5.48, 0.73, 4.08, 1.88, 5.31, 0.78, 4.36, 1.71, 5.62, 0.43, 4.21, 1.21]
# [w14, w24, w34, w15 .. w68, w78]
weights = [-0.1, 0.2, 0.1, 0.2, 0.5, -0.7, 0.1, -0.3, 0.5, 0.3, 0.6, -0.4, 0.3, 0.3, 0.2, 0.1]

nn = NeuralNetwork(weights)
nn.train()
