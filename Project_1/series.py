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
    def __init__(self, weights_=None, iterations=1000, lmd=0.1, input_size=3, hidden_size=4, output_size=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.iterations = iterations
        self.lmd = lmd
        self.conn_n = get_connections_n(self.hidden_size, self.input_size, self.output_size)
        self.weights = weights_ if isinstance(weights_, list) and len(weights_) == self.conn_n else np.random.randn(self.conn_n)
        self.hidden_neurones = []

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
        weights_ = self.get_hidden_weights()
        self.hidden_neurones = [Neuron(weights_[i], f'Neuron {self.input_size + i + 1}') for i in range(len(weights_))]

    def get_hidden_fS(self, inputs):
        for neuron in self.hidden_neurones:
            neuron.set_neuron_fS(neuron.forward(inputs))
        return [neuron.forward(inputs) for neuron in self.hidden_neurones]

    def get_out_fS(self, hidden_fS):
        return f(np.dot(hidden_fS, self.weights[self.conn_n - self.hidden_size:]))

    # def get_adjusted_weights(self, delta):
    #     for neuron in self.hidden_neurones:
    #         neuron.set_delta(delta * self.lmd * df(neuron))

    # def forward(self, inputs):
     #     sum = np.dot(self.get_hidden_weights(), inputs)


    def train(self):
        # for iteration in 2000:
        for iteration in range(self.iterations):
            self.initialize_hidden()
            inputs = [2.65, 5.60, 1.21]
            output_weights = self.weights[self.conn_n - self.hidden_size:]
            hidden_fS = self.get_hidden_fS(inputs)
            out_fS = self.get_out_fS(hidden_fS)  #############
            # error = 5.48 - out_fS  ##########################
            error = out_fS - 5.48  ##########################
            delta = error * df(out_fS)

            # get adjusted weights for hidden
            for i in range(len(self.hidden_neurones)):
                neuron = self.hidden_neurones[i]
                # print(output_weights[i], "  -->  ", neuron.neuron_fS)
                neuron.set_delta(delta * output_weights[i] * df(neuron.neuron_fS))

            res = []
            for i in self.hidden_neurones:
                delta_w = i.delta * self.lmd * np.array(inputs)
                result_w = np.array(i.weights) - delta_w
                res = np.concatenate([res, result_w])
                # print(i, '\n', 'Δw', delta_w, '\n', '‾w', result_w, '\n')

            # get adjusted weights for output
            delta_w_out = delta * self.lmd * np.array(hidden_fS)
            result_w_out = np.array(output_weights) - delta_w_out
            # print('         Δw', delta_w_out)
            # print('         ‾w', result_w_out)
            #
            # # result = res + result_w_out
            # print(np.concatenate([res, result_w_out]))
            self.weights = np.concatenate([res, result_w_out])

        self.initialize_hidden()
        inputs = [2.65, 5.60, 1.21]
        output_weights = self.weights[self.conn_n - self.hidden_size:]
        hidden_fS = self.get_hidden_fS(inputs)
        out_fS = self.get_out_fS(hidden_fS)
        print(self.hidden_neurones)
        print(out_fS)




data = [2.65, 5.60, 1.21, 5.48, 0.73, 4.08, 1.88, 5.31, 0.78, 4.36, 1.71, 5.62, 0.43, 4.21, 1.21]
# [w14, w15, w16, w17, w24 ...]
# weights = [-0.1, 0.2, 0.1, 0.3, 0.1, 0.5, -0.3, 0.6, 0.3, -0.7, 0.5, -0.4, 0.3, 0.3, 0.2, 0.1]
# [w14, w24, w34, .. ? .. ]
weights = [-0.1, 0.2, 0.1, 0.2, 0.5, -0.7, 0.1, -0.3, 0.5, 0.3, 0.6, -0.4, 0.3, 0.3, 0.2, 0.1]

nn = NeuralNetwork(weights)
nn.train()
# print(nn.get_hidden_weights())







# w = np.array_split([-0.1, 0.2, 0.1, 0.2, 0.5, -0.7, 0.1, -0.3, 0.5, 0.3, 0.6, -0.4], 3)
# print(w)
# train_data = np.array([[data[ind], data[ind + 1], data[ind + 2]] for ind in range(10)])
# print(f(np.dot(f(np.dot(train_data, w)), [0.3, 0.3, 0.2, 0.1])))
#










# nn.train()

# test = np.array([[-0.1 , 0.2 , 0.1 , 0.2], [ 0.5 ,-0.7 , 0.1, -0.3], [ 0.5 , 0.3 , 0.6 ,-0.4]])
# W1 = np.array([[-0.2, 0.3, -0.4], [0.1, -0.3, -0.4]])
# print(np.array(np.array_split([-0.1, 0.2, 0.1, 0.2, 0.5, -0.7, 0.1, -0.3, 0.5, 0.3, 0.6, -0.4], 3)))
# print(np.dot(W1, [2.65, 5.60, 1.21]))
# print(np.dot(test, [2.65, 5.60, 1.21]))
# training_input_data = [[0.13, 5.97, 0.57],
#  [5.97, 0.57, 4.02],
#  [0.57, 4.02 ,0.31],
#  [4.02, 0.31, 5.55],
#  [0.31, 5.55 ,0.15],
#  [5.55 ,0.15, 4.54],
#  [0.15, 4.54 ,0.65],
#  [4.54 ,0.65 ,4.34],
#  [0.65 ,4.34 ,1.54],
#  [4.34 ,1.54 ,4.7 ]]
# all_weights1= [[0.13014705, 0.60179093 ,0.54086457, 0.58055869],
#  [0.84098176, 0.63216505, 0.38153196, 0.65285245],
#  [0.81100167 ,0.70115784, 0.09563249 ,0.12707542]]
# print(np.dot(training_input_data, all_weights1))

# W1 = np.array([-0.2, 0.3, -0.4])
# x = [1, 10, 100]
# print(W1[:] * x[0:3])
#
# print(0.06472316611941606 * 0.1 * 2.65)

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
