import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    # x - preaktywacje
    # d/dx relu(x) = 1 jeśli x>0, else 0
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
    # y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))
    # return -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))

def binary_cross_entropy_derivative(y_pred, y_true):
    epsilon = 1e-8
    # y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (y_pred - y_true) / ((y_pred * (1 - y_pred)) + epsilon)
    # return (y_pred - y_true) / ((y_pred * (1 - y_pred)) + epsilon)

def wasserstein_loss_disc(real_output, fake_output):
    # Loss for the discriminator
    return -np.mean(real_output) + np.mean(fake_output)

def wasserstein_loss_gen(fake_output):
    # Loss for the generator
    return -np.mean(fake_output)

def wasserstein_loss_disc_derivative(real_output, fake_output):
    # Gradienty dla rzeczywistych i fałszywych danych
    grad_real = -np.ones_like(real_output)
    grad_fake = np.ones_like(fake_output)
    return grad_real, grad_fake

def wasserstein_loss_gen_derivative(fake_output):
    # Gradient dla generatora
    return -np.ones_like(fake_output)

def compute_gradient_penalty(discriminator, real_samples, fake_samples, lambda_gp=10):
    # Interpolate between real and fake samples
    alpha = np.random.uniform(0, 1, size=real_samples.shape)
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples

    # Forward pass through the discriminator
    interpolates_output = discriminator.forward(interpolates)

    # Compute gradients with respect to inputs
    gradients = np.gradient(interpolates_output, interpolates, edge_order=2)
    gradient_norms = np.sqrt(np.sum(gradients**2, axis=(1, 2, 3)))

    # Gradient penalty
    gradient_penalty = np.mean((gradient_norms - 1)**2)
    return lambda_gp * gradient_penalty


def initialize_weights(shape):
    return np.random.randn(*shape) * 0.01

def xavier_initialization(shape):
    n_in, n_out = shape
    limit = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(-limit, limit, shape)

def he_initialization(shape):
    n_in, n_out = shape
    return np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)