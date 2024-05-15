import torch


class CrossEntropyLoss:

    def __init__(self):
        self.input = None
        self.label = None

    def __call__(self, predicted, label):
        self.input = predicted
        self.label = label
        return (-label * torch.log(predicted)).reshape(label.shape[0], 1, -1).sum(dim=-1)

    def backward(self):
        grad = -self.label / self.input
        self.input = None
        self.label = None
        return grad
