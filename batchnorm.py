import numpy as np

class BatchNorm2d:
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        self.gamma = np.ones((1, num_features, 1, 1))
        self.beta = np.zeros((1, num_features, 1, 1))
        self.running_mean = np.zeros((1, num_features, 1, 1))
        self.running_var = np.ones((1, num_features, 1, 1))

        self.cache = {}

    def forward(self, x, training=True):
        N, C, H, W = x.shape

        if training:
            mean = np.mean(x, axis=(0, 2, 3), keepdims=True)  # (1, C, 1, 1)
            var = np.var(x, axis=(0, 2, 3), keepdims=True)    # (1, C, 1, 1)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var  = self.momentum * self.running_var  + (1 - self.momentum) * var
        else:
            mean = self.running_mean
            var  = self.running_var

        x_hat = (x - mean) / np.sqrt(var + self.eps)

        out = self.gamma * x_hat + self.beta

        self.cache = {
            'x_hat': x_hat,
            'mean': mean,
            'var': var,
            'x': x,
            'gamma': self.gamma,
        }
        return out

    def backward(self, dout):
        x_hat = self.cache['x_hat']
        mean = self.cache['mean']
        var = self.cache['var']
        x = self.cache['x']
        gamma = self.cache['gamma']
        eps = self.eps

        N, C, H, W = dout.shape
        size = N * H * W

        # dBeta = suma po wszystkich wymiarach oprócz C
        dbeta = np.sum(dout, axis=(0,2,3), keepdims=True)  # (1, C, 1, 1)
        # dGamma = suma z (dout * x_hat)
        dgamma = np.sum(dout * x_hat, axis=(0,2,3), keepdims=True)

        # dx_hat
        dx_hat = dout * gamma  # (N, C, H, W)

        # dvar = suma( dx_hat * (x - mean) * -1/2 * (var+eps)^(-3/2) )
        dvar = np.sum(dx_hat * (x - mean), axis=(0,2,3), keepdims=True) * (-0.5) * (var + eps)**(-3/2)

        # dmean
        dmean = np.sum(dx_hat * -1 / np.sqrt(var + eps), axis=(0,2,3), keepdims=True)
        dmean += dvar * np.mean(-2.0 * (x - mean), axis=(0,2,3), keepdims=True)

        # dx
        dx = dx_hat / np.sqrt(var + eps)
        dx += dvar * 2.0 * (x - mean) / size
        dx += dmean / size

        return dx, dgamma, dbeta
