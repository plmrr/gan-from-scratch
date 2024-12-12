import numpy as np
from utils import xavier_initialization, leaky_relu, sigmoid, leaky_relu_derivative, sigmoid_derivative, he_initialization
from utils import tanh, tanh_derivative
from conv import Conv2D

class Discriminator:
    def __init__(self):
        # kernel 4x4
        # stride 2
        # padding 1
        self.conv1 = Conv2D(3, 64, 4, 2, 1)    # 64 16 16
        self.conv2 = Conv2D(64,128,4,2,1)      # 128 8 8 
        self.conv3 = Conv2D(128,256,4,2,1)     # 256 4 4

        # FC
        scale = 0.02
        self.W_fc = np.random.randn(256*4*4,1)*scale
        self.b_fc = np.zeros((1,1))

        self.cache = {}

    def forward(self, x):
        # x: (N,3,32,32)
        out1 = self.conv1.forward(x)
        self.cache['out1_pre'] = out1
        out1 = leaky_relu(out1, 0.2)

        out2 = self.conv2.forward(out1)
        self.cache['out2_pre'] = out2
        out2 = leaky_relu(out2, 0.2)

        out3 = self.conv3.forward(out2)
        self.cache['out3_pre'] = out3
        out3 = leaky_relu(out3,0.2)

        out3_flat = out3.reshape(x.shape[0], -1)
        self.cache['out3_flat'] = out3_flat

        logits = out3_flat.dot(self.W_fc) + self.b_fc
        self.cache['logits'] = logits
        out = sigmoid(logits)
        self.cache['out'] = out
        return out

    def backward(self, dout):
        # dout = dL/d(sigmoid)
        logits = self.cache['logits']
        dsig = dout * sigmoid_derivative(logits)

        out3_flat = self.cache['out3_flat']
        dW_fc = out3_flat.T.dot(dsig)
        db_fc = np.sum(dsig, axis=0, keepdims=True)

        dflat = dsig.dot(self.W_fc.T)
        # reshape do (N,256,4,4)
        N = out3_flat.shape[0]
        dconv3_out = dflat.reshape(N,256,4,4)

        out3_pre = self.cache['out3_pre']
        dconv3_out_pre = dconv3_out * leaky_relu_derivative(out3_pre,0.2)
        dx3, dW3, db3 = self.conv3.backward(dconv3_out_pre)

        out2_pre = self.cache['out2_pre']
        dconv2_out = dx3 * leaky_relu_derivative(out2_pre,0.2)
        dx2, dW2, db2 = self.conv2.backward(dconv2_out)

        out1_pre = self.cache['out1_pre']
        dconv1_out = dx2 * leaky_relu_derivative(out1_pre,0.2)
        dx1, dW1, db1 = self.conv1.backward(dconv1_out)

        return dx1, (dW_fc, db_fc, dW1, db1, dW2, db2, dW3, db3)
