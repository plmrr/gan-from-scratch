import numpy as np
from utils import initialize_weights

class Discriminator:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.weights = []
        self.biases = []
        layers_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layers_sizes)-1):
            w = initialize_weights((layers_sizes[i], layers_sizes[i+1]))
            b = np.zeros((1, layers_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self):
        # placeholder
        pass

    def backward(self):
        # placeholder
        pass