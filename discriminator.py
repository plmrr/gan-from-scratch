import numpy as np
from utils import xavier_initialization, leaky_relu, sigmoid, leaky_relu_derivative, sigmoid_derivative

class Discriminator:
    def __init__(self, input_size, hidden_sizes):
        self.weights = []
        self.biases = []
        layers_sizes = [input_size] + hidden_sizes + [1] # output is 1 value
        for i in range(len(layers_sizes) - 1):
            w = xavier_initialization((layers_sizes[i], layers_sizes[i + 1]))
            b = np.zeros((1, layers_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        self.activations = [x]  # post-aktywacje
        self.pre_activations = []  # pre-aktywacje
        for i in range(len(self.weights)):
            pre_activation = np.dot(x, self.weights[i]) + self.biases[i]
            self.pre_activations.append(pre_activation)
            if i == len(self.weights) - 1:
                # output layer: sigmoid
                x = sigmoid(pre_activation)
            else:
                # hidden layers: leaky relu
                x = leaky_relu(pre_activation)
            self.activations.append(x)
        return x

    def backward(self, grad_output):
        grads_w = []
        grads_b = []
        grad_input = grad_output * sigmoid_derivative(self.pre_activations[-1])
        
        for i in reversed(range(len(self.weights))):
            # print(f"layer {i}:")
            # print(f"grad_input.shape = {grad_input.shape}")
            # print(f"self.activations[{i}].shape = {self.activations[i].shape}")
            grad_w = np.dot(self.activations[i].T, grad_input) / grad_input.shape[0]
            grad_b = np.sum(grad_input, axis=0, keepdims=True) / grad_input.shape[0]
            grads_w.insert(0, grad_w)
            grads_b.insert(0, grad_b)
            
            if i > 0:
                grad_input = np.dot(grad_input, self.weights[i].T)
                grad_input *= leaky_relu_derivative(self.pre_activations[i - 1])
            else:
                grad_input = np.dot(grad_input, self.weights[i].T)
        # print(f"Desc backward function returns: grads_w ({len(grads_w)}), grads_b ({len(grads_b)}), grad_input ({grad_input.shape})")
        return grads_w, grads_b, grad_input

    def classify(self, images):
        return self.forward(images)
