import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from ..builder import HEADS
from .cls_head import ClsHead


@HEADS.register_module()
class AMSoftmaxClsHead(ClsHead):
    """AMSoftmax classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        scale (int): scale with normalized cosine scores.
        margin (float): margin of AmSoftmax
        loss (dict): Config of classification loss.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 scale=30,
                 margin=0.2,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk=(1, )):
        super(AMSoftmaxClsHead, self).__init__(loss=loss, topk=topk)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.s = scale
        self.m = margin

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
        # print(x)
        # compute cosine linear
        cosine = self.cosine_sim(x, self.W)
        # label mapping
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, gt_label.view(-1, 1), 1.0)
        cls_score = self.s * (cosine - one_hot * self.m)
        losses = self.loss(cls_score, gt_label)
        return losses
