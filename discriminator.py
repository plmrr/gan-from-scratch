import numpy as np
from utils import leaky_relu, sigmoid, leaky_relu_derivative, sigmoid_derivative
from conv import Conv2D
from batchnorm import BatchNorm2d

class Discriminator:
    def __init__(self):
        # kernel 4x4, stride 2, pad=1
        self.conv1 = Conv2D(3,   64, 4, 2, 1)   # out: (64,16,16)
        self.conv2 = Conv2D(64, 128, 4, 2, 1)   # out: (128,8,8)
        self.conv3 = Conv2D(128,256, 4, 2, 1)   # out: (256,4,4)
        self.bn2 = BatchNorm2d(128)
        self.bn3 = BatchNorm2d(256)

        # FC
        scale = 0.02
        self.W_fc = np.random.randn(256*4*4,1)*scale
        self.b_fc = np.zeros((1,1))

        self.cache = {}

    def forward(self, x, training=True):
        # x: (N,3,32,32)
        out1 = self.conv1.forward(x)              
        self.cache['out1_conv'] = out1
        out1 = leaky_relu(out1, 0.2)

        out2 = self.conv2.forward(out1)           # (N,128,8,8)
        self.cache['out2_conv'] = out2
        # BN2 -> LeakyReLU
        out2_bn = self.bn2.forward(out2, training=training)
        self.cache['out2_bn'] = out2_bn
        out2_act = leaky_relu(out2_bn, 0.2)
        self.cache['out2_act'] = out2_act

        out3 = self.conv3.forward(out2_act)       # (N,256,4,4)
        self.cache['out3_conv'] = out3
        # BN3 -> LeakyReLU
        out3_bn = self.bn3.forward(out3, training=training)
        self.cache['out3_bn'] = out3_bn
        out3_act = leaky_relu(out3_bn, 0.2)
        self.cache['out3_act'] = out3_act

        out3_flat = out3_act.reshape(x.shape[0], -1)
        self.cache['out3_flat'] = out3_flat

        logits = out3_flat.dot(self.W_fc) + self.b_fc
        self.cache['logits'] = logits
        out = sigmoid(logits)
        self.cache['out'] = out
        return out

    def backward(self, dout):
        # dout = dL/d(sigmoid_output)
        logits = self.cache['logits']
        dsig = dout * sigmoid_derivative(logits)  # (N,1)

        out3_flat = self.cache['out3_flat']       # (N,256*4*4)
        dW_fc = out3_flat.T.dot(dsig)             # (4096,1)
        db_fc = np.sum(dsig, axis=0, keepdims=True)

        dflat = dsig.dot(self.W_fc.T)             # (N,4096)
        dconv3_out = dflat.reshape(-1,256,4,4)    # (N,256,4,4)

        # WARSTWA 3: LeakyReLU -> BN3 -> Conv3
        out3_bn = self.cache['out3_bn']           # (N,256,4,4)  PRZED LeakyReLU
        out3_conv = self.cache['out3_conv']       # (N,256,4,4)  PRZED BN3

        # d(LeakyReLU)
        d_lrelu3 = dconv3_out * leaky_relu_derivative(out3_bn, 0.2)  # (N,256,4,4)

        # d(BN3)
        dx_bn3, dgamma3, dbeta3 = self.bn3.backward(d_lrelu3)

        # d(Conv3)
        dx3, dW3, db3 = self.conv3.backward(dx_bn3)

        # WARSTWA 2: LeakyReLU -> BN2 -> Conv2
        out2_bn  = self.cache['out2_bn']   # PRZED LeakyReLU
        out2_conv = self.cache['out2_conv']

        d_lrelu2 = dx3 * leaky_relu_derivative(out2_bn, 0.2)
        dx_bn2, dgamma2, dbeta2 = self.bn2.backward(d_lrelu2)

        dx2, dW2, db2 = self.conv2.backward(dx_bn2)

        # WARSTWA 1: LeakyReLU -> Conv1 (brak BN1 w tym przykładzie)
        out1_conv = self.cache['out1_conv']  # PRZED LeakyReLU
        d_lrelu1 = dx2 * leaky_relu_derivative(out1_conv, 0.2)
        dx1, dW1, db1 = self.conv1.backward(d_lrelu1)

        # [W_fc, b_fc, W1, b1, W2, b2, W3, b3, gamma2, beta2, gamma3, beta3]
        return dx1, (dW_fc, db_fc,
                     dW1, db1,
                     dW2, db2,
                     dW3, db3,
                     dgamma2, dbeta2,
                     dgamma3, dbeta3)
