import numpy as np


# Define activation function
def sigmoid(x):
    return 1 if x >= 0.5 else 0
    # return 1 / (1 + np.exp(-x))


# Define artificial neuron class
class Neuron:
    def __init__(self, weights):
        self.weights = weights

    def forward(self, inputs):
        total = np.dot(self.weights, inputs)
        # print('S = ', total)
        return sigmoid(total)


# Define neural network class
class NeuralNetwork:
    def __init__(self):
        self.neuron1 = Neuron([1, -1])
        self.neuron2 = Neuron([-1, 1])
        self.neuron3 = Neuron([1, 1])

    def forward(self, input1, input2):
        output1 = self.neuron1.forward([input1, input2])
        output2 = self.neuron2.forward([input1, input2])
        output3 = self.neuron3.forward([output1, output2])
        return output3


# Create neural network object
nn = NeuralNetwork()

# Test the neural network with inputs 0 and 0
output = nn.forward(0, 0)
print(output)  # Expected output: 0

# Test the neural network with inputs 0 and 1
output = nn.forward(0, 1)
print(output)  # Expected output: 1

# Test the neural network with inputs 1 and 0
output = nn.forward(1, 0)
print(output)  # Expected output: 1

# Test the neural network with inputs 1 and 1
output = nn.forward(1, 1)
print(output)  # Expected output: 0
