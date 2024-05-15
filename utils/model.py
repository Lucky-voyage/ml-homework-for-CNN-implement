import torch
import json
from layers.AvgPooling import AvgPooling
from layers.Conv2d import Conv2d
from layers.DepthWise import DepthWiseConv
from layers.Linear import Linear
from layers.Relu import Relu


class Model:

    def __init__(self, classes):
        self.classes = classes
        self.input = None

        self.conv1 = Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.dw_conv2 = DepthWiseConv(8, 16, 1)
        self.dw_conv3 = DepthWiseConv(16, 32, 1)
        self.dw_conv4 = DepthWiseConv(32, 64, 1)
        self.fc = Linear(64, classes)
        self.relu = Relu()
        self.avg_pooling = AvgPooling()

    def forward(self, X: torch.tensor) -> torch.tensor:
        X = self.conv1(X)
        X = self.relu(X)
        X = self.dw_conv2(X)
        X = self.relu(X)
        X = self.dw_conv3(X)
        X = self.relu(X)
        X = self.dw_conv4(X)
        X = self.relu(X)
        X = self.avg_pooling(X)
        X = self.relu(X)
        X = X.view(X.size(0), -1)
        X = self.fc(X)
        return X

    def __call__(self, X: torch.tensor) -> torch.tensor:
        return self.forward(X)

    def zero_grad(self):
        self.input = None
        self.relu.zero_grad()
        self.avg_pooling.zero_grad()

    def backward(self, partial, lr=0.001):
        partial = self.fc.backward(partial, lr)
        partial = self.relu.backward(partial)
        partial = self.avg_pooling.backward(partial)
        partial = self.relu.backward(partial)
        partial = self.dw_conv4.backward(partial, lr)
        partial = self.relu.backward(partial)
        partial = self.dw_conv3.backward(partial, lr)
        partial = self.relu.backward(partial)
        partial = self.dw_conv2.backward(partial, lr)
        partial = self.relu.backward(partial)
        self.conv1.backward(partial, lr)

    def save(self, path):
        checkpoint = self.parameters()
        with open(str(path), 'w') as f:
            json.dump(checkpoint, f)

    def parameters(self):
        return {
            "conv1": self.conv1.parameters(),
            "dw_conv2": self.dw_conv2.parameters(),
            "dw_conv3": self.dw_conv3.parameters(),
            "dw_conv4": self.dw_conv4.parameters(),
            "fc": self.fc.parameters()
        }

    def load(self, params):
        try:
            return ((self.conv1.load(params["conv1"]) and self.dw_conv2.load(params["dw_conv2"])
                    and self.dw_conv3.load(params["dw_conv3"]) and self.dw_conv4.load(params["dw_conv4"])) and
                    self.fc.load(params["fc"]))
        except:
            return False
