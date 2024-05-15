import torch


class AvgPooling:

    def __init__(self):
        self.inputs = []

    def __call__(self, X):
        # X shape (batch, c, h, w)
        if X.shape.__len__() != 4:
            X.unsqueeze_(0)
        self.inputs.append(X)
        return X.mean(dim=(2, 3), keepdim=True)

    def zero_grad(self):
        self.inputs = []

    def backward(self, partial):
        # partial shape: batch output_shape
        # return shape:  batch input_shape
        mask = torch.ones_like(self.inputs.pop())
        total = float(mask.shape[-1] * mask.shape[-2])
        return mask / total

