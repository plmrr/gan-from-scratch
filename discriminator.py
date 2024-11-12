import numpy as np
from utils import initialize_weights, leaky_relu, sigmoid

class Discriminator:
    def __init__(self, input_size, hidden_sizes):
        self.weights = []
        self.biases = []
        layers_sizes = [input_size] + hidden_sizes + [1] # output is 1 value
        for i in range(len(layers_sizes) - 1):
            w = initialize_weights((layers_sizes[i], layers_sizes[i + 1]))
            b = np.zeros((1, layers_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x):
        self.activations = [x]
        x = x.reshape(x.shape[0], -1)
        for i in range(len(self.weights) - 1):
            x = np.dot(x, self.weights[i]) + self.biases[i]
            x = leaky_relu(x)
            self.activations.append(x)
        
        x = np.dot(x, self.weights[-1]) + self.biases[-1]
        x = sigmoid(x)
        self.activations.append(x)
        return x

    def backward(self):
        # placeholder
        pass