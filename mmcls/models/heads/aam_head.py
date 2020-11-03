import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from ..builder import HEADS
from .cls_head import ClsHead
import math

@HEADS.register_module()
class AAMSoftmaxClsHead(ClsHead):
    """AAMSoftmax classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        scale (int): scale with normalized cosine scores.
        margin (float): margin of AmSoftmax
        loss (dict): Config of classification loss.
        easy_margin (bool) : whether do easy margin
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 scale=30,
                 margin=0.2,
                 easy_margin=False,
                 max_iteration=80000,
                 start_ratio=0.5,
                 margin_freq=10,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk=(1, )):
        super(AAMSoftmaxClsHead, self).__init__(loss=loss, topk=topk)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.s = scale
        self._iter = 0
        self.freq = margin_freq
        self.max_iteration = max_iteration
        self.start_ratio = start_ratio
        self.max_margin = margin
        self.m = 0
        self.easy_margin = easy_margin

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self._init_layers()

    def _init_layers(self):
        self.W = nn.Parameter(torch.randn(self.num_classes, self.in_channels))

    def cosine_sim(self, x1, x2, dim=1, eps=1e-8):
        # print(x1, x2)
        ip = torch.mm(x1, x2.T)
        w1 = torch.norm(x1, 2, dim)
        w2 = torch.norm(x2, 2, dim)
        return ip / torch.ger(w1, w2).clamp(min=eps)

    def _margin_schedule(self):
        iter = min(self._iter, self.max_iteration)
        process = max(
            (iter - self.max_iteration * self.start_ratio)/(self.max_iteration*(1-self.start_ratio)),
            0)
        process = (process - 1)*math.pi
        process = (math.cos(process) + 1) / 2
        self.m = process * self.max_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def init_weights(self):
        nn.init.xavier_uniform_(self.W)

    def simple_test(self, img):
        """Test without augmentation."""
        cls_score = self.s * self.cosine_sim(img, self.W)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        if torch.onnx.is_in_onnx_export():
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred

    def forward_train(self, x, gt_label):
        self._iter += 1
        if self._iter % self.freq == 0:
            self._margin_schedule()
        # print(x)
        # compute cosine linear
        cosine = self.cosine_sim(x, self.W)

        # cos(theta+m)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, gt_label.view(-1, 1), 1)
        cls_score = self.s * ((one_hot * phi) + ((1.0 - one_hot) * cosine))
        loss = self.loss(cls_score, gt_label)
        return loss