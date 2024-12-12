import numpy as np

def im2col(images, filter_h, filter_w, stride=1, pad=0):
    # images: (N, C, H, W)
    N, C, H, W = images.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    
    img_padded = np.pad(images, [(0,0), (0,0), (pad,pad), (pad,pad)], mode='constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img_padded[:, :, y:y_max:stride, x:x_max:stride]
    col = col.transpose(0,4,5,1,2,3).reshape(N*out_h*out_w, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    # input_shape: (N, C, H, W)
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0,3,4,5,1,2)
    
    img = np.zeros((N, C, H+2*pad, W+2*pad))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
    return img[:, :, pad:H+pad, pad:W+pad]

class Conv2D:
    def __init__(self, in_channels, out_channels, filter_size=3, stride=1, pad=1):
        scale = np.sqrt(2.0/(in_channels*filter_size*filter_size))
        self.W = np.random.randn(out_channels, in_channels, filter_size, filter_size)*scale
        self.b = np.zeros((out_channels,))
        self.stride = stride
        self.pad = pad
        self.cache = None

    def forward(self, x):
        # x: (N,C,H,W)
        N, C, H, W = x.shape
        out_h = (H + 2*self.pad - self.W.shape[2])//self.stride + 1
        out_w = (W + 2*self.pad - self.W.shape[3])//self.stride + 1
        
        col = im2col(x, self.W.shape[2], self.W.shape[3], self.stride, self.pad)
        W_col = self.W.reshape(self.W.shape[0], -1)
        
        out = np.dot(col, W_col.T) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0,3,1,2) # (N, out_channels, out_h, out_w)
        
        self.cache = (x, col, W_col)
        return out

    def backward(self, dout):
        # dout: (N, out_channels, out_h, out_w)
        x, col, W_col = self.cache
        N, C, H, W = x.shape
        out_channels = self.W.shape[0]
        
        dout_reshaped = dout.transpose(0,2,3,1).reshape(-1, out_channels)
        dW = np.dot(dout_reshaped.T, col)
        dW = dW.reshape(self.W.shape)
        
        db = np.sum(dout_reshaped, axis=0)
        
        dcol = np.dot(dout_reshaped, W_col)
        dx = col2im(dcol, x.shape, self.W.shape[2], self.W.shape[3], self.stride, self.pad)
        
        return dx, dW, db

class ConvTranspose2D:
    def __init__(self, in_channels, out_channels, filter_size=4, stride=2, pad=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad

        fan_in = in_channels * filter_size * filter_size
        scale = np.sqrt(2.0/fan_in)
        self.W = np.random.randn(in_channels, out_channels, filter_size, filter_size)*scale
        self.b = np.zeros((out_channels,))
        self.cache = None

    def forward(self, x):
        N, C, H, W = x.shape
        filter_h, filter_w = self.filter_size, self.filter_size

        H_out = (H - 1)*self.stride - 2*self.pad + filter_h
        W_out = (W - 1)*self.stride - 2*self.pad + filter_w

        H_up = (H - 1)*self.stride + 1
        W_up = (W - 1)*self.stride + 1
        x_up = np.zeros((N, C, H_up, W_up), dtype=x.dtype)
        x_up[:, :, ::self.stride, ::self.stride] = x

        pad_conv = filter_h - 1 - self.pad

        # im2col
        col = im2col(x_up, filter_h, filter_w, stride=1, pad=pad_conv)
        W_col = self.W.reshape(self.in_channels*filter_h*filter_w, self.out_channels)

        out = np.dot(col, W_col) + self.b
        out = out.reshape(N, H_out, W_out, self.out_channels).transpose(0,3,1,2)

        self.cache = (x, x_up, col, W_col, H_out, W_out, pad_conv)
        return out

    def backward(self, dout):
        x, x_up, col, W_col, H_out, W_out, pad_conv = self.cache
        N, C, H, W = x.shape
        filter_h, filter_w = self.filter_size, self.filter_size

        # dout: (N, out_channels, H_out, W_out)
        dout_reshaped = dout.transpose(0,2,3,1).reshape(-1, self.out_channels) # (N*H_out*W_out, out_channels)

        # dW
        dW = np.dot(col.T, dout_reshaped)
        dW = dW.reshape(self.in_channels, filter_h, filter_w, self.out_channels)
        dW = dW.transpose(0,3,1,2)

        # db
        db = np.sum(dout_reshaped, axis=0) # (out_channels,)

        # dx_up
        dcol = np.dot(dout_reshaped, W_col.T)  # (N*H_out*W_out, C*filter_h*filter_w)
        dx_up = col2im(dcol, x_up.shape, filter_h, filter_w, stride=1, pad=pad_conv)

        dx = np.zeros_like(x)
        dx[:, :, :, :] = dx_up[:, :, ::self.stride, ::self.stride]

        return dx, dW, db