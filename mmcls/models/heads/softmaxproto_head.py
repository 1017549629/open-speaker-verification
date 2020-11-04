import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from ..builder import HEADS
from .cls_head import ClsHead


@HEADS.register_module()
class SoftmaxProtoHead(ClsHead):
    """Softmax Proto classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 init_w=10.0,
                 init_b=-5.0,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk=(1, )):
        super(SoftmaxProtoHead, self).__init__(loss=loss, topk=topk)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.init_w = init_w
        self.init_b = init_b

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self._init_layers()

    def _init_layers(self):
        self.fc_softmax = nn.Linear(self.in_channels, self.num_classes)
        self.W_angleproto = nn.Parameter(torch.tensor(self.init_w))
        self.b_angleproto = nn.Parameter(torch.tensor(self.init_b))
        self.criterion_angleproto = nn.CrossEntropyLoss()

    def init_weights(self):
        normal_init(self.fc_softmax, mean=0, std=0.01, bias=0)

    def simple_test(self, img):
        """Test without augmentation."""
        cls_score = self.fc_softmax(img)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        if torch.onnx.is_in_onnx_export():
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred

    def compute_angle_loss(self, x):
        # Thanks to ClovaAI voxceleb_trainer angleproto loss
        out_anchor = torch.mean(x[:, 1:, :], 1)
        out_positive = x[:, 0, :]
        spkr_size = out_positive.shape[0]

        cos_sim_mat = F.cosine_similarity(out_positive.unsqueeze(-1), out_anchor.unsqueeze(-1).transpose(0, 2))
        torch.clamp(self.W_angleproto, 1e-6)
        cos_sim_mat = cos_sim_mat * self.W_angleproto + self.b_angleproto
        gt_label = torch.arange(0, spkr_size).to(x.device).long()
        return self.criterion_angleproto(cos_sim_mat, gt_label)

    def forward_train(self, x, gt_label):
        cls_score = self.fc_softmax(x)
        loss_s = self.loss(cls_score, gt_label)
        # APloss is sampling with N_spkr*2 batchsize
        ap_x = x.reshape(-1, 2, x.shape[-1])
        loss_ap = self.compute_angle_loss(ap_x)
        loss_s["loss"] += loss_ap
        return loss_s
