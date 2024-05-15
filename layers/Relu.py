import torch


class Relu:

    def __init__(self):
        self.grad = []

    def __call__(self, x):
        self.grad.append(torch.where(x > 0, x, 0))
        return torch.where(x > 0, x, 0)

    def zero_grad(self):
        self.grad = []

    def backward(self, partial):
        # partial shape: batch, output_features
        # return shape:  batch, input_features
        grad = torch.where(self.grad.pop() > 0, 1, 0)
        return grad * partial
