import numpy as np
from utils import xavier_initialization, leaky_relu, tanh, tanh_derivative, leaky_relu_derivative

class Generator:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.weights = []
        self.biases = []
        layers_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layers_sizes) - 1):
            w = xavier_initialization((layers_sizes[i], layers_sizes[i + 1]))
            b = np.zeros((1, layers_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, z):
        self.activations = [z]  # post-aktywacje
        self.pre_activations = []  # pre-aktywacje
        x = z
        for i in range(len(self.weights)):
            pre_activation = np.dot(x, self.weights[i]) + self.biases[i]
            self.pre_activations.append(pre_activation)
            if i == len(self.weights) - 1:
                # output layer: tanh
                x = tanh(pre_activation)
            else:
                # hidden layers: leaky relu
                x = leaky_relu(pre_activation)
            self.activations.append(x)
        return x

    def backward(self, grad_output):
        grads_w = []
        grads_b = []
        grad_input = grad_output * tanh_derivative(self.pre_activations[-1])
        
        for i in reversed(range(len(self.weights))):
            grad_w = np.dot(self.activations[i].T, grad_input) / grad_input.shape[0]
            grad_b = np.sum(grad_input, axis=0, keepdims=True) / grad_input.shape[0]
            grads_w.insert(0, grad_w)
            grads_b.insert(0, grad_b)
            
            if i > 0:
                grad_input = np.dot(grad_input, self.weights[i].T)
                grad_input *= leaky_relu_derivative(self.pre_activations[i - 1])
            else:
                grad_input = np.dot(grad_input, self.weights[i].T)
        return grads_w, grads_b
    
    def generate(self, batch_size):
        z = np.random.randn(batch_size, self.weights[0].shape[0])
        return self.forward(z)
