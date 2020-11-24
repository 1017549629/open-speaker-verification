import mmcv
from mmcls.models import build_classifier
from mmcv.runner import load_checkpoint
from argparse import ArgumentParser
import torch
import kaldiio
import os
import numpy as np


parser = ArgumentParser("feach_fc")
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--target", type=str, required=True)
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()


@torch.no_grad()
def load_model(checkpoint, config):
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    model = build_classifier(config.model)
    checkpoint = load_checkpoint(model, checkpoint, map_location="cpu")
    model.cfg = config  # save the config in the model for convenience
    model.eval()
    return model

def normalize(mat):
    mat_norm = np.linalg.norm(mat, axis=1).reshape([mat.shape[0], 1])
    mat = mat / mat_norm
    return mat


def main():
    model = load_model(args.ckpt, args.config)
    speaker_data = model.head.W.data.cpu().numpy()
    speaker_data = normalize(speaker_data)
    print(np.matmul(speaker_data, speaker_data.T))
    spk1 = np.matmul(speaker_data, speaker_data.T)[1]
    k = []
    for i in spk1:
        if i > 0.25:
            k.append(i)
    print(k)
    print(len(k))
    # spkr_size = speaker_data.shape[0]
    # os.makedirs(args.target, exist_ok=True)
    # spkr_cohort_scp = os.path.join(args.target, "spkr_cohort.scp")
    # spkr_cohort_ark = os.path.join(args.target, "spkr_cohort.ark")
    # with kaldiio.WriteHelper("ark,scp:%s,%s" % (spkr_cohort_ark, spkr_cohort_scp)) as f:
    #     for i in range(spkr_size):
    #         vector = speaker_data[i]
    #         # print(vector.shape)
    #         f("spkr_cohort_%d" % i, vector)


main()