import numpy as np
from utils import relu, relu_derivative, tanh, tanh_derivative
from conv import ConvTranspose2D
from batchnorm import BatchNorm2d

class Generator:
    def __init__(self, noise_dim=100):
        self.noise_dim = noise_dim
        # FC: noise_dim -> 512*4*4
        scale = 0.02
        self.W_fc = np.random.randn(noise_dim, 512*4*4)*scale
        self.b_fc = np.random.randn(1, 512*4*4) * scale
        #self.b_fc = np.zeros((1, 512*4*4))

        # Deconv
        self.deconv1 = ConvTranspose2D(512, 256, 4, 2, 1)  # (N, 512, 4,4) -> (N,256,8,8)
        self.deconv2 = ConvTranspose2D(256, 128, 4, 2, 1)  # -> (N,128,16,16)
        self.deconv3 = ConvTranspose2D(128, 3,   4, 2, 1)  # -> (N,3,32,32)

        # BATCHNORM – do warstw z ReLU
        self.bn1 = BatchNorm2d(256)
        self.bn2 = BatchNorm2d(128)

        self.cache = {}

    def forward(self, z, training=True):
        # FC
        fc_out = z.dot(self.W_fc) + self.b_fc
        fc_out_reshaped = fc_out.reshape(z.shape[0], 512, 4, 4)
        
        # Deconv1
        out1 = self.deconv1.forward(fc_out_reshaped)   # (N,256,8,8)
        out1 = self.bn1.forward(out1, training=training)
        out1 = relu(out1)

        # Deconv2
        out2 = self.deconv2.forward(out1)              # (N,128,16,16)
        out2 = self.bn2.forward(out2, training=training)
        out2 = relu(out2)

        # Deconv3
        out3 = self.deconv3.forward(out2)              # (N,3,32,32)
        out3 = tanh(out3)

        # Zapis w cache do backward
        self.cache['z'] = z
        self.cache['fc_out'] = fc_out
        self.cache['out1_pre'] = out1
        self.cache['out2_pre'] = out2
        self.cache['out3_pre'] = out3    # tanh input
        return out3

    def backward(self, dout):
        # dout -> d(tanh(out3_pre)) = dout * (1 - tanh^2)
        out3_pre = self.cache['out3_pre']
        dpre3 = dout * tanh_derivative(out3_pre)

        # Deconv3
        dx3, dW3, db3 = self.deconv3.backward(dpre3)

        out2_pre = self.cache['out2_pre'] 
        drelu2 = dx3 * relu_derivative(out2_pre)
        dx_bn2, dgamma2, dbeta2 = self.bn2.backward(drelu2)
        dx2, dW2, db2 = self.deconv2.backward(dx_bn2)

        # BN1 backward => analogicznie
        out1_pre = self.cache['out1_pre']  # output BN (przed ReLU)
        drelu1 = dx2 * relu_derivative(out1_pre)
        dx_bn1, dgamma1, dbeta1 = self.bn1.backward(drelu1)
        dx1, dW1, db1 = self.deconv1.backward(dx_bn1)

        # FC backward
        z = self.cache['z']
        fc_out = self.cache['fc_out']  # (N, 512*4*4)
        dfc = dx1.reshape(z.shape[0], 512*4*4)
        dW_fc = z.T.dot(dfc)
        db_fc = np.sum(dfc, axis=0, keepdims=True)
        dZ = dfc.dot(self.W_fc.T)

        return dZ, (dW_fc, db_fc, dW1, db1, dW2, db2, dW3, db3,
                    dgamma1, dbeta1, dgamma2, dbeta2)
