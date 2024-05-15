import torch
import torch.nn.functional as F
from utils.Transpose_Conv import TransposeConv
from layers.Conv2d import Conv2d


class DepthWise:

    def __init__(self,
                 channels,
                 padding,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 device='cpu'):
        self.c = channels
        self.stride = stride
        self.kh = kernel_size[0]
        self.kw = kernel_size[1]
        if not isinstance(stride, tuple):
            stride = (stride, stride)
        self.sh = stride[0]
        self.sw = stride[1]
        self.padding = padding
        self.device = device

        self.input = None
        self.transpose_conv = TransposeConv(self.c, self.c, kernel_size, bias=True, padding=padding, stride=stride)

        self.W = torch.rand(self.c, self.kh, self.kw).to(self.device)
        self.bias = torch.rand(self.c).to(self.device)

    def __call__(self, X):
        if X.shape.__len__() != 4:
            X.unsqueeze_(0)
            batch = 1
        else:
            batch = X.shape[0]
        self.input = X

        # W shape (batch, c, kh * kw)
        # X_new shape (batch, c, kh * kw, oh * ow)
        W = self.W.repeat(batch, 1, 1, 1).reshape(batch, self.c, -1)

        oh = int((X.shape[2] + self.padding * 2 - self.kh) / self.sh + 1)
        ow = int((X.shape[3] + self.padding * 2 - self.kw) / self.sw + 1)

        X_new = torch.ones(batch, self.c, self.kh * self.kw, oh * ow)
        for i in range(self.c):
            X_new[:, i, :, :] = F.unfold(X[:, i, :, :].unsqueeze(1), (self.kh, self.kw),
                                         stride=(self.sh, self.sw), padding=self.padding)

        outputs = (torch.einsum('bcm,bcmn->bcn', W, X_new) +
                   self.bias.repeat(batch, int(X_new.shape[-1]), 1).permute(0, 2, 1))

        return outputs.reshape(batch, self.c, oh, ow)

    def load(self, params):
        try:
            if (params['weight'].numel() == self.W.numel() and
                    (params['bias'] is None or params['bias'].shape == self.bias.shape)):
                self.W = params['weight'].detach().reshape(tuple(self.W.shape))
                self.bias = params['bias'].detach()
                return True
            return False
        except KeyError:
            return False

    def backward(self, partial, learning_rate):
        grad_X = self.transpose_conv.gradForInput(self.input, partial, self.W, groups=True)
        grad_W = self.transpose_conv.gradForWeight(self.input, partial, self.W, groups=True)
        grad_bias = self.transpose_conv.gradForBias(partial)

        self.W -= learning_rate * grad_W
        self.bias -= learning_rate * grad_bias
        return grad_X

    def parameters(self):
        return {
            'weight': self.W,
            'bias': self.bias
        }


class DepthWiseConv:

    def __init__(self, in_channels, out_channels, stride=(1, 1)):
        self.depth_wise = DepthWise(in_channels, kernel_size=(3, 3), stride=stride, padding=1)
        self.point_wise = Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def __call__(self, X):
        X = self.depth_wise(X)
        X = self.point_wise(X)
        return X

    def backward(self, partial, lr):
        partial = self.point_wise.backward(partial, lr)
        return self.depth_wise.backward(partial, lr)

    def parameters(self):
        return {
            'depth_wise': self.depth_wise.parameters(),
            'point_wise': self.point_wise.parameters()
        }

    def load(self, params):
        try:
            return (self.depth_wise.load(params['depth_wise']) and
                    self.point_wise.load(params['point_wise']))
        except KeyError:
            return False


import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


