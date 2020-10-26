import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer
from mmcls.models.utils import SELayer1D as SELayer


class TDNN_pad(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride=1,
                 with_se=False, se_reduction=16, se_bias=False,
                 dilation=1, batch_norm=True, pad=True,
                 nonlinearity="LeakyReLU"):
        super(TDNN_pad, self).__init__()
        self.context_size = kernel_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.pad = pad
        if self.pad:
            self.pad_length = (kernel_size-1) * dilation // 2
        self.with_se = with_se
        if self.with_se:
            self.se = SELayer(self.output_dim, reduction=se_reduction,
                              nonlinearity=nonlinearity, bias=se_bias)

        self.kernel = nn.Conv1d(self.input_dim, self.output_dim, self.context_size,
                                dilation=self.dilation, stride=self.stride)
        self.nonlinearity = build_activation_layer(dict(type=nonlinearity, inplace=True))
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        if self.pad:
            x = nn.functional.pad(x, (self.pad_length, self.pad_length))
        x = self.kernel(x)
        if self.with_se:
            x = self.se(x)

        x = self.nonlinearity(x)

        if self.batch_norm:
                x = self.bn(x)
        return x


class RES_TDNN_SE(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, dilation=1, se_reduction=16,
                 se_bias=False, nonlinearity="LeakyReLU", bias=False):
        super(RES_TDNN_SE, self).__init__()
        self.do_transform = input_dim != output_dim
        if self.do_transform:
            self.transform = nn.Sequential(
                nn.Conv1d(input_dim, output_dim, 1, bias=False),
                nn.BatchNorm1d(output_dim)
            )
        pad = (kernel_size - 1) * dilation // 2
        self.seqs = nn.Sequential(
            TDNN_pad(input_dim, output_dim, 1),
            nn.Conv1d(output_dim, output_dim, kernel_size, dilation=dilation, padding=pad, bias=bias),
            SELayer(output_dim, reduction=se_reduction, bias=se_bias)
        )
        self.post_shortcut = nn.Sequential(
            build_activation_layer(dict(type=nonlinearity, inplace=True)),
            nn.BatchNorm1d(output_dim)
        )

    def forward(self, x):
        y = self.seqs(x)
        if self.do_transform:
            return self.post_shortcut(self.transform(x) + y)
        else:
            return self.post_shortcut(x+y)


if __name__ == "__main__":
    # tdnn module test
    tdnn = TDNN_pad(128, 256, 3, with_se=True, se_reduction=8)
    print(tdnn)
    data = torch.rand(2, 128, 81)
    print(tdnn(data).shape)

    # res_tdnn_se module test
    tdnn_res = RES_TDNN_SE(128, 128, 5, se_bias=True)
    print(tdnn_res)
    print(tdnn_res(data).shape)