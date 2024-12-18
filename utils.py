import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(x.dtype)

def leaky_relu(x, alpha=0.02):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.02):
    return np.where(x > 0, 1, alpha)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def binary_cross_entropy(y_pred, y_true):
    epsilon = 1e-8
    return -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))

def binary_cross_entropy_derivative(y_pred, y_true):
    epsilon = 1e-8
    return (y_pred - y_true) / ((y_pred * (1 - y_pred)) + epsilon)

def initialize_weights(shape):
    return np.random.randn(*shape) * 0.01

def xavier_initialization(shape):
    n_in, n_out = shape
    limit = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(-limit, limit, shape)

def he_initialization(shape):
    n_in, n_out = shape
    return np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)