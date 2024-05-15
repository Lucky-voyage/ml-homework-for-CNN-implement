import torch
import torch.nn.functional as F


class TransposeConv:

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding=(0, 0), stride=(1, 1)):
        self.ic = in_channels
        self.oc = out_channels
        self.stride = stride
        self.bias = bias

        if not isinstance(kernel_size, tuple):
            self.kh = kernel_size
            self.kw = kernel_size
        else:
            self.kh = kernel_size[0]
            self.kw = kernel_size[1]
        if not isinstance(padding, tuple):
            self.ph = padding
            self.pw = padding
        else:
            self.ph = padding[0]
            self.pw = padding[1]

    def gradForInput(self, inputs, partial, kernel, groups=False):
        # partial shape (batch, oc, oh, ow)
        # kernel shape  (oc, ic * kh * kw)

        # flip kernel, k shape (ic, oc * kh * kw)
        if not groups:
            k = kernel.reshape(self.oc, self.ic, self.kh, self.kw)
            k = torch.flip(k, dims=(-2, -1)).permute(1, 0, 2, 3).reshape(self.ic, -1)
        else:
            # for each channel, there is a corresponding kernel with shape (1, ic, kh, kw)
            # Due to there are 'oc' groups, the shape for kernel is (oc, ic, kh, kw)
            # Here we un_squeeze kernel to shape (oc, 1, ic * kh * kw), and slice kernel[i] for each channel
            k = kernel.reshape(self.oc, 1, -1)

        # calculate the paddings for partial
        # partial_pad shape (batch, oc, oh, ow)
        oh = int((inputs.shape[-2] + 2 * self.ph - self.kh) / self.stride[0] + 1)
        ow = int((inputs.shape[-1] + 2 * self.pw - self.kw) / self.stride[1] + 1)
        ph = int((self.stride[0] + 1) * (inputs.shape[-2] - oh) / 2 + self.ph * self.stride[0])
        pw = int((self.stride[1] + 1) * (inputs.shape[-1] - ow) / 2 + self.pw * self.stride[1])

        if not groups:
            # shape update_kernel (batch, oc, ih * iw)
            update = self.calc_conv(k.repeat(inputs.shape[0], 1, 1),
                                    partial, (self.kh, self.kw), self.stride, (ph, pw))
        else:
            update = torch.ones_like(inputs)
            # partial shape (oc, batch, oh, ow)
            partial = partial.reshape(partial.shape[0], self.oc, oh, ow).permute(1, 0, 2, 3)
            for i in range(self.oc):
                update[:, i, :, :] = self.calc_conv(k[i].unsqueeze(0), partial[i].unsqueeze(1),
                                                    (self.kh, self.kw), self.stride,
                                                    (ph, pw)).reshape(1, inputs.shape[-2], inputs.shape[-1])

        return update

    def gradForWeight(self, inputs, partial, kernel, groups=False):
        if not groups:
            # partial shape (batch, oc, oh, ow)
            # p shape (batch, oc, ic * oh * ow)
            # input shape (batch, ic, ih, iw)
            p = partial.unsqueeze(2).repeat_interleave(self.ic, dim=2).reshape(inputs.shape[0], self.oc, -1)

            grad_W = torch.zeros(inputs.shape[0], self.oc, self.kh * self.kw * self.ic)
            for i in range(inputs.shape[0]):
                # p_new shape (1, oc, oh * ow)
                input_new = inputs[i].unsqueeze(0)
                p_new = p[i].unsqueeze(0)
                grad_W[i] = self.calc_conv(p_new, input_new, kernel_size=(partial.shape[-2], partial.shape[-1]),
                                           stride=self.stride, padding=(self.ph, self.pw))
        else:
            batch = inputs.shape[0]
            kh = int((inputs.shape[-2] + 2 * self.ph - self.kh) / self.stride[0] + 1)
            kw = int((inputs.shape[-1] + 2 * self.pw - self.kw) / self.stride[1] + 1)
            partial = partial.reshape(batch, self.oc, -1)
            grad_W = torch.zeros(inputs.shape[0], self.oc, self.kh * self.kw)
            for i in range(batch):
                p = partial[i]
                ip = inputs[i]
                for j in range(self.oc):
                    p_new = p[j].reshape(1, 1, -1)
                    input_new = ip[j].unsqueeze(0).unsqueeze(0)
                    grad_W[i, j] = self.calc_conv(p_new, input_new, (kh, kw), self.stride,
                                                  (self.ph, self.pw)).reshape(9)
            grad_W = grad_W.reshape(self.oc, self.kh, self.kw)
        return grad_W.mean(0)

    def gradForBias(self, partial):
        # partial shape (batch, oc, oh, ow)
        return partial.mean(dim=0).reshape(self.oc, -1).sum(dim=-1)

    @staticmethod
    def calc_conv(X, Y, kernel_size, stride, padding):
        Y_new = F.unfold(Y, kernel_size=kernel_size, stride=stride, padding=padding)
        return torch.einsum("bcm,bmn->bcn", X, Y_new)

    @staticmethod
    def padding(inputs, pad=(0, 0)):
        ph = pad[0]
        pw = pad[1]
        ih = inputs.shape[-2]
        iw = inputs.shape[-1]

        mask = torch.zeros(ih + 2 * ph, iw + 2 * pw)
        mask[:, :, ph:ph + ih, ph:ph + iw] = input
        return mask
