import torch


class Linear:

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device='cpu'):
        self.inf = in_features
        self.otf = out_features
        self.W = torch.rand((out_features, in_features)).to(device)
        self.device = device

        if bias:
            self.bias = torch.zeros(out_features)
        else:
            self.bias = None

        self.input = None

    def __call__(self, X):
        shape = list(X.shape)

        # calculate batch
        if len(shape) == 1:
            X = X.unsqueeze(0)
            batch = 1
        else:
            batch = shape[0]
        X_new = X.unsqueeze(2)
        self.input = X_new

        W = self.W.repeat(batch, 1, 1)
        outputs = torch.matmul(W, X_new)

        if self.bias is not None:
            outputs += self.bias.unsqueeze(1).repeat(batch, 1, 1)

        return outputs

    def backward(self, partial, learning_rate):
        W = self.W

        # partial_output / partial_input
        # partial shape (batch, output_feature, 1)
        # X shape (batch, input_feature, 1)
        # W shape (output_feature, input_feature)
        grad = torch.matmul(W.permute(1, 0), partial)

        # calculate grad_W
        # mask shape (batch, otf, itf) -> (otf, itf)
        mask = torch.matmul(partial, self.input.permute(0, 2, 1))
        self.W -= learning_rate * mask.mean(0)

        # calculate grad_bias
        # partial shape (batch, output_feature, 1)
        if self.bias is not None:
            self.bias -= learning_rate * partial.mean(0).reshape(-1)

        return grad

    def parameters(self):
        return {
            'weight': self.W,
            'bias': self.bias
        }

    def load(self, params):
        try:
            if (params['weight'].numel() == self.W.numel() and
                    (params['bias'] is None or params['bias'].numel() == self.otf)):
                self.W = params['weight'].reshape(tuple(self.W.shape))
                self.bias = params['bias']
                return True

            return False
        except KeyError:
            return False



