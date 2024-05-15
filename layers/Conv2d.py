from utils.Transpose_Conv import TransposeConv
import multiprocessing
import torch.nn.functional as F
import torch


def process_task(i, j, stride, kh, kw, batch, X, X_new):
    start_i = i * stride
    start_j = j * stride
    X_new[:, :, i: i + 1, j: j + 1] = (
        X[:, :, start_i: start_i + kh, start_j: start_j + kw].reshape(batch, -1, 1, 1))


class Conv2d:

    def __init__(self, input_channels,
                 output_channels,
                 kernel_size,
                 bias=True,
                 stride=(1, 1),
                 padding=1,
                 device='cpu'):

        self.ic = input_channels
        self.oc = output_channels
        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size)
        self.kh = kernel_size[0]
        self.kw = kernel_size[1]
        if not isinstance(stride, tuple):
            stride = (stride, stride)
        self.sh = stride[0]
        self.sw = stride[1]
        self.stride = stride
        self.padding = padding
        self.device = device

        self.input = None
        self.transpose_conv = TransposeConv(self.ic, self.oc, kernel_size, bias=bias, stride=stride, padding=padding)

        # weights -- hyperparameters
        # use 'expand' for batch operation
        self.W = torch.rand(self.oc, self.ic * self.kh * self.kw).to(self.device).to(torch.float32)

        if bias:
            self.bias = torch.rand(self.oc).to(self.device).to(torch.float32)
        else:
            self.bias = None

        # calculate number of processes
        if device == 'cpu':
            self.processes = max(multiprocessing.cpu_count() - 2, 0)

    def __call__(self, X):
        """
            X:  input, after padding
                shape: (batch_size, intput_channels, height, width)
            self.W:
                shape: (output_channels, input_channels, k_height, k_width)

            For each x in batch X:
                Firstly, flatten the W to shape: (output_channels, input_channels * k_height * k_width)

                According to the definition of Conv2D, the layer will retrieve a part of x, denoted by x_i with
                shape (input_channels, k_height, k_width). Therefore, we will slice X to small pieces and
                get (kh * kw) pieces.
                    kh = (height - k_height) / stride + 1
                    kw = (width - k_width) / stride + 1

                The output of x_i, we denote it to y_i, has shape (output_channels, 1)
                To accelerate the calculation, we can flatten x_i to shape (1, input_channels * k_height * k_width)
                So, def y_i = f(x_i):
                        return W * transpose(x_i)

                Last, we need to stack all of y_i to generate Y. The shape of Y is (output_channels, kh, kw)
                For getting the answer with shape before, we need a reshaped X, which has shape
                (inputs_channels * k_height * k_width, kh, kw)

                Finally, def F(X):
                            # X.shape = (batch_size, input_channels, height, width)
                            # W.shape = (output_channels, input_channels, k_height, k_width)

                            X = X -> (inputs_channels * k_height * k_width, kh, kw)
                            W = W.reshape(output_channels, input_channels * k_height * k_width)
                            return W * X
        """
        # 1st dim of X is batch_size
        shape = list(X.shape)
        if X.shape.__len__() != 4:
            X = X.unsqueeze(0)
            batch = 1
        else:
            batch = shape[0]
        self.input = X

        ic = shape[-3]
        ih = shape[-2]
        iw = shape[-1]
        oh = int((ih + self.padding * 2 - self.kh) / self.sh + 1)
        ow = int((iw + self.padding * 2 - self.kw) / self.sw + 1)

        # use for block
        # calculate the output for 100,000 times need 42s
        # but torch.nn.Conv2d just consume 2.5s

        # shape (batch, ic * kh * kw, oh, ow)
        # X_new = torch.zeros((batch, ic * self.kh * self.kw, oh, ow),
        #                     dtype=torch.float32, device=self.device)
        # for i in range(oh):
        #     start_i = i * self.sh
        #     for j in range(ow):
        #         start_j = j * self.sw
        #         # get small piece X_piece with shape (batch, ic * kh * kw, 1)
        #         X_piece = X[:, :, start_i: start_i + self.kh, start_j: start_j + self.kw].reshape(self.batch, -1, 1, 1)
        #         X_new[:, :, i: i + 1, j: j + 1] = X_piece

        # use F.unfold
        # calculate the output for 100,000 times need 7.9s
        # but torch.nn.Conv2d just consume 2.5s

        # X_new = F.unfold(X, kernel_size=(self.kh, self.kw), stride=self.stride).reshape(batch, -1, oh, ow)
        # self.unfold = X_new

        # use multiple processes
        # shape (batch, ic * kh * kw, oh, ow)
        # X_new = torch.zeros((batch, ic * self.kh * self.kw, oh, ow),
        #                     dtype=torch.float32, device=self.device)
        # if self.device == 'cpu':
        #     tasks = [process_task(i, j, self.stride, self.kh, self.kw, self.batch, X, X_new)
        #              for i in range(oh) for j in range(ow)]
        #     with Pool(processes=self.processes) as pool:
        #         pool.map(process_task, tasks)

        # calculate the grad
        # ignoring batch, K shape (oc, ic * kh * kw), X shape(ic * kh * kw, oh * ow)
        # O = K * X
        # /frac{/partial{O}}{/partial{K}} = X^T
        # grad shape (batch, oh * ow, ic * kh * kw)

        # X  (batch, ic, ih, iw)
        # W  (batch, oc, ic * kh * kw)

        # W     shape (batch, oc, ic * kh * kw)
        # X_new shape (batch, ic * kh * kw, oh * ow)
        W = self.W.repeat(batch, 1, 1)
        outputs = self.calc_conv(W, X, (self.kh, self.kw), (self.sh, self.sw), self.padding)

        if self.bias is not None:
            outputs += self.bias.repeat(batch, oh * ow, 1).permute(0, 2, 1)

        return outputs.reshape(batch, -1, oh, ow)

    def backward(self, partial, learning_rate):
        grad_X = self.transpose_conv.gradForInput(self.input, partial, self.W, groups=False)
        grad_W = self.transpose_conv.gradForWeight(self.input, partial, self.W, groups=False)
        self.W -= learning_rate * grad_W

        if self.bias is not None:
            self.bias -= learning_rate * self.transpose_conv.gradForBias(partial)

        return grad_X

    @staticmethod
    def calc_conv(X, Y, kernel_size, stride, padding):
        Y_new = F.unfold(Y, kernel_size=kernel_size, stride=stride, padding=padding)
        return torch.einsum("bcm,bmn->bcn", X, Y_new)

    def parameters(self):
        if self.bias:
            return {
                'weight': self.W,
                'bias': self.bias
            }
        else:
            return {
                'weight': self.W,
                'bias': None
            }

    def load(self, params):
        try:
            if (params['weight'].numel() == self.W.numel() and
                    (params['bias'] is None or params['bias'].numel() == self.oc)):
                self.W = params['weight'].detach().reshape(tuple(self.W.shape))
                self.bias = params['bias'].detach()
                return True

            return False
        except KeyError:
            return False



