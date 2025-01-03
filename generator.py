import numpy as np
from utils import relu, relu_derivative, tanh, tanh_derivative
from conv import ConvTranspose2D
class Generator:
    def __init__(self, noise_dim=100):
        self.noise_dim = noise_dim
        # FC: noise_dim -> 512*4*4
        scale = 0.02
        self.W_fc = np.random.randn(noise_dim, 512*4*4)*scale
        self.b_fc = np.random.randn(1, 512*4*4) * scale

        # kernel 4x4
        # stride 2
        # padding 1
        # ConvTranspose warstwy
        self.deconv1 = ConvTranspose2D(512, 256, 4, 2, 1) # 256 8 8 
        self.deconv2 = ConvTranspose2D(256, 128, 4, 2, 1) # 128 16 16
        self.deconv3 = ConvTranspose2D(128, 3, 4, 2, 1) # 3 32 32

        self.cache = {}

    def forward(self, z):
        # FC
        fc_out = z.dot(self.W_fc) + self.b_fc
        fc_out_reshaped = fc_out.reshape(z.shape[0], 512, 4, 4)
        self.cache['fc_out'] = fc_out
        self.cache['z'] = z

        # deconv1
        out1 = self.deconv1.forward(fc_out_reshaped)
        # ReLU
        self.cache['out1_pre'] = out1
        out1 = relu(out1)

        # deconv2
        out2 = self.deconv2.forward(out1)
        self.cache['out2_pre'] = out2
        out2 = relu(out2)

        # deconv3
        out3 = self.deconv3.forward(out2)
        self.cache['out3_pre'] = out3
        out3 = tanh(out3)

        self.cache['out1_act'] = out1
        self.cache['out2_act'] = out2
        self.cache['out3_act'] = out3
        return out3

    def backward(self, dout):
        # dout = dL/d(G_out), gdzie G_out = tanh(out3_pre)
        # dtanh/dx = 1 - tanh^2(x)
        out3_pre = self.cache['out3_pre']
        dpre3 = dout * tanh_derivative(out3_pre)

        dx3, dW3, db3 = self.deconv3.backward(dpre3)

        # out2_act = relu(out2_pre)
        out2_pre = self.cache['out2_pre']
        dout2 = dx3 * relu_derivative(out2_pre)
        dx2, dW2, db2 = self.deconv2.backward(dout2)

        # out1_act = relu(out1_pre)
        out1_pre = self.cache['out1_pre']
        dout1 = dx2 * relu_derivative(out1_pre)
        dx1, dW1, db1 = self.deconv1.backward(dout1)

        # FC backward
        z = self.cache['z']
        fc_out = self.cache['fc_out'] # (N, 512*4*4)
        dfc = dx1.reshape(z.shape[0], 512*4*4)
        dW_fc = z.T.dot(dfc)
        db_fc = np.sum(dfc, axis=0, keepdims=True)
        dZ = dfc.dot(self.W_fc.T)

        return dZ, (dW_fc, db_fc, dW1, db1, dW2, db2, dW3, db3)
