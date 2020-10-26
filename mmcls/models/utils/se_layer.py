import mmcv
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule
from mmcv.cnn import build_activation_layer


class SELayer(nn.Module):
    """Squeeze-and-Excitation Module.

    Args:
        channels (int): The input (and output) channels of the SE layer.
        ratio (int): Squeeze ratio in SELayer, the intermediate channel will be
            ``int(channels/ratio)``. Default: 16.
        conv_cfg (None or dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        act_cfg (dict or Sequence[dict]): Config dict for activation layer.
            If act_cfg is a dict, two activation layers will be configurated
            by this dict. If act_cfg is a sequence of dicts, the first
            activation layer will be configurated by the first dict and the
            second activation layer will be configurated by the second dict.
            Default: (dict(type='ReLU'), dict(type='Sigmoid'))
    """

    def __init__(self,
                 channels,
                 ratio=16,
                 conv_cfg=None,
                 act_cfg=(dict(type='ReLU'), dict(type='Sigmoid'))):
        super(SELayer, self).__init__()
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert mmcv.is_tuple_of(act_cfg, dict)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=int(channels / ratio),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=int(channels / ratio),
            out_channels=channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out


class SELayer1D(nn.Module):
    """
    Some theory tells that SE layer using conv1d or conv2d with kernel size as 1 is better than
    just using linear layer
    """
    def __init__(self, channel, reduction=16, nonlinearity="LeakyReLU", bias=False):
        super(SELayer1D, self).__init__()
        in_channel = channel
        self.fc = nn.Sequential(
            nn.Conv1d(in_channel, channel//reduction, kernel_size=1, bias=bias),
            build_activation_layer(dict(type=nonlinearity, inplace=True)),
            nn.Conv1d(channel//reduction, channel, kernel_size=1, bias=bias),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch, channel, _ = x.size()
        y = torch.mean(x, -1)
        y = y.unsqueeze(-1)
        y = self.fc(y)
        return x*y.expand_as(x)


if __name__ == "__main__":
    se1d = SELayer1D(128, nonlinearity="ReLU")
    data = torch.rand(2, 128, 56)
    print(se1d)
    print(se1d(data).shape)