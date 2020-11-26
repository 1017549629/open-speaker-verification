import torch
import torch.nn as nn
from mmcls.models.utils import RES_TDNN_SE as RES_TDNN
from mmcls.models.utils import TDNN_pad as TDNN
from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from mmcv.cnn import (constant_init, kaiming_init)


@BACKBONES.register_module()
class RES_SE_TDNN(BaseBackbone):
    def __init__(self, in_dim, in_channel=512, res_layers=[2, 2, 2],
                 output_size=1500, se_reduction=8, se_bias=True, bn_momentum=0.5):
        super(RES_SE_TDNN, self).__init__()
        layers = [TDNN(in_dim, in_channel, kernel_size=5,
                       with_se=True, se_reduction=se_reduction,
                       se_bias=se_bias, bn_momentum=bn_momentum)]
        self.context_sizes = [5, 7, 1]
        for kernel, res_num in zip(self.context_sizes, res_layers):
            for i in range(res_num):
                layers.append(RES_TDNN(in_channel, in_channel, kernel, bn_momentum=bn_momentum,
                                       se_reduction=se_reduction, se_bias=se_bias))
        layers.append(TDNN(in_channel, output_size, kernel_size=1, with_se=True,
                           se_reduction=se_reduction, se_bias=se_bias, bn_momentum=bn_momentum))
        self.layers = nn.Sequential(*layers)
        self.out_dim = output_size

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                constant_init(m.weight, 1)
                constant_init(m.bias, 0)

    def forward(self, inp):
        if isinstance(inp, tuple):
            x, label = inp
        else:
            x, label = inp, None
        x = x.transpose(-1, -2)
        x = self.layers(x)
        return x