import torch
import torch.nn as nn

from ..builder import NECKS
from mmcv.cnn import build_norm_layer


def select_activation(activation_type):
    if activation_type == "leaky_relu":
        return nn.LeakyReLU(inplace=True)
    elif activation_type == "relu":
        return nn.ReLU(inplace=True)
    elif activation_type == "prelu":
        return nn.PReLU()
    elif activation_type == "none":
        return nn.Identity()
    else:
        print("activation type {} is not supported".format(activation_type))
        raise NotImplementedError


def std_pooling(batch, batch_mean, dim=-1, unbiased=False, eps=1e-8):
    # adding epsilon in sqrt function to make more numerically stable results (yufeng)
    r2 = torch.sum((batch - batch_mean.unsqueeze(-1))**2, dim)
    if unbiased:
        length = batch.shape[dim] - 1
    else:
        length = batch.shape[dim]
    return torch.sqrt(r2/length + eps)


class Stats_pooling(nn.Module):
    def __init__(self, input_dim=1500):
        super(Stats_pooling, self).__init__()
        self.out_dim = 2 * input_dim

    def forward(self, x):
        """
        x.size() = [batch_size, feature_dim, seq_length]
        """
        mean_frame = torch.mean(x, -1, False)
        if self.training:
            std_frame = std_pooling(x, mean_frame, -1, False)
        else:
            std_frame = torch.std(x, -1, False)
        output = torch.cat([mean_frame, std_frame], dim=-1)
        # print(output.shape)
        output = output.view(-1, self.out_dim)
        return output


@NECKS.register_module()
class StatsPooling(nn.Module):
    """Stats Pooling neck.
    """
    def __init__(self, in_plane, emb_dim, emb_bn=True, emb_affine=True, activation_type="relu", norm_type="BN1d"):
        super(StatsPooling, self).__init__()
        self.avgpool = Stats_pooling(in_plane)
        embedding = []
        initial_dim = self.avgpool.out_dim
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
        pass

    def forward(self, inputs):
        out = self.avgpool(inputs)
        return self.embedding(out)