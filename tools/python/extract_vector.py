import mmcv
from mmcls.models import build_classifier
from mmcv.runner import load_checkpoint
from tqdm import tqdm
from argparse import ArgumentParser
import torch
from multiprocessing.pool import Pool
import os
import kaldiio
import numpy as np
import random

parser = ArgumentParser("extract_xvector")
parser.add_argument("--feat_scp", type=str, required=True)
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--target", type=str, required=True)
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--do_plda", action="store_true")
parser.add_argument("--nj", type=int, default=4)
args = parser.parse_args()


def load_model(checkpoint, config):
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    model = build_classifier(config.model)
    checkpoint = load_checkpoint(model, checkpoint)
    model.cfg = config  # save the config in the model for convenience
    if args.do_plda:
        print("---INFO---: Far layer selected for plda")
        model.neck.embedding[1] = torch.nn.Identity()
    model.eval()
    return model


def divide_feats(feats_scp, nj):
    utt_arks = []
    with open(feats_scp) as f:
        for line in f:
            utt_arks.append(line.strip().split(" "))
    random.shuffle(utt_arks)
    sub_process = []
    length = len(utt_arks)
    step = length // nj + 1
    for i in range(step):
        sub_process.append(utt_arks[i*step: i*step+step])
    return sub_process


@torch.no_grad()
def extract_xvector(process, pid, target, ckpt=args.ckpt):
    os.makedirs(target, exist_ok=True)
    xvector_scp = os.path.join(target, "%d_vector.scp" % pid)
    xvector_ark = os.path.join(target, "%d_vector.ark" % pid)
    writer = kaldiio.WriteHelper

    cuda_id = pid % 4
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_id)
    model = load_model(ckpt, args.config)
    model.cuda()
    with writer("ark,scp:%s,%s" % (xvector_ark, xvector_scp)) as f:
        if pid == 0:
            iterator = tqdm(process)
        else:
            iterator = process
        for utt, rxfile in iterator:
            feat = torch.from_numpy(kaldiio.load_mat(rxfile).astype(np.float32)).cuda()
            xvector = model.extract_feat(feat.unsqueeze(0))
            xvector = xvector.cpu().numpy()
            f(utt, xvector[0])


def main():
    sub_process = divide_feats(args.feat_scp, args.nj)
    # pool = Pool(args.nj)
    # for i in range(args.nj):
    #     pool.apply_async(extract_xvector, (sub_process[i], i, args.target, args.ckpt,))
    # pool.close()
    # pool.join()
    extract_xvector(sub_process[0], 1, args.target, args.ckpt)

main()