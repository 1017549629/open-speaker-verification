import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import NECKS
from mmcv.cnn import build_norm_layer
from .STP import select_activation


@NECKS.register_module()
class ECAPA_ASP(nn.Module):
    """
    ECAPA Attentive Statistics pooling
    """
    def __init__(self, in_plane, att_dim=128, att_with_context=True, central_bn=True, emb_dim=192, emb_bn=False,
                 emb_affine=False, activation_type="none", norm_type="BN1d"):
        super(ECAPA_ASP, self).__init__()
        self.W1 = nn.Linear(in_plane, att_dim)
        self.att_with_context = att_with_context
        if att_with_context:
            self.W2 = nn.Linear(att_dim, in_plane)
        else:
            self.W2 = nn.Linear(att_dim, 1)
        self.act = select_activation("relu")
        self.central_bn = central_bn
        if self.central_bn:
            self.pool_bn = build_norm_layer(dict(type=norm_type, momentum=0.5), in_plane*2)[1]
        embedding = []
        initial_dim = in_plane * 2
        if isinstance(emb_dim, list):
            for e_dim, do_bn, do_affine, act_type in zip(emb_dim, emb_bn, emb_affine, activation_type):
                fc = [nn.Linear(initial_dim, e_dim)]
                initial_dim = e_dim
                fc.append(select_activation(act_type))
                if do_bn:
                    cfg = dict(type=norm_type, requires_grad=True, momentum=0.5, affine=do_affine)
                    fc.append(build_norm_layer(cfg, e_dim)[1])
                embedding.append(nn.Sequential(*fc))
        else:
            embedding.append(nn.Linear(initial_dim, emb_dim))
            embedding.append(select_activation(activation_type))
            if emb_bn:
                cfg = dict(type=norm_type, requires_grad=True, momentum=0.5, affine=emb_affine)
                embedding.append(build_norm_layer(cfg, emb_dim)[1])
        self.embedding = nn.Sequential(*embedding)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data)

    def forward(self, x):
        # B,C,T to B,T,C
        x = x.transpose(1, 2)
        attn = self.W2(self.act(self.W1(x)))
        attn = F.softmax(attn, 1)
        # print(attn.shape, x.shape)
        mean = torch.sum(torch.mul(attn, x), 1)
        astd = torch.sqrt(torch.clamp(torch.sum(torch.mul(attn, x**2), 1) - mean**2, min=1e-8))
        out = torch.cat((mean, astd), 1)
        if self.central_bn:
            out = self.pool_bn(out)

        # print(out.shape)
        return self.embedding(out)