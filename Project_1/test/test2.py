import numpy as np


# Define sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Define neural network class with backpropagation
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)

    def forward(self, X):
        self.hidden = sigmoid(np.dot(X, self.weights1))
        self.output = sigmoid(np.dot(self.hidden, self.weights2))
        return self.output

    def backward(self, X, y, learning_rate):
        error = y - self.output
        d_output = error * sigmoid_derivative(self.output)
        error_hidden = d_output.dot(self.weights2.T)
        d_hidden = error_hidden * sigmoid_derivative(self.hidden)
        self.weights2 += self.hidden.T.dot(d_output) * learning_rate
        self.weights1 += X.T.dot(d_hidden) * learning_rate

    def train(self, X, y, learning_rate, num_epochs):
        for i in range(num_epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)

    def predict(self, X):
        return self.forward(X)


# Define input and output data
X = np.array([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9]])
y = np.array([[0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9], [1.0]])

# Create neural network object and train it
nn = NeuralNetwork(1, 4, 1)
nn.train(X, y, 0.1, 10000)

# Predict the output for a new input
X_new = np.array([[1.0]])
y_new = nn.predict(X_new)
print(y_new)  # Expected output: [[1.044]] (approx.)
