# created by Yufeng Ma
# email: wyrmyf@gmail.com

import mmcv
import numpy as np

from .builder import DATASETS
from .base_dataset import BaseDataset
import kaldiio
import os
import json
import random
import copy


def get_spk2utt(file):
    spk2utt = []
    with open(file, "r") as f:
        for line in f.readlines():
            line = line.strip()
            spk, utt_list = line.split()[0], line.split()[1:]
            spk2utt.append((spk, utt_list))
    return spk2utt


def get_utt2spk(file):
    utt2spk = {}
    with open(file, "r") as f:
        for line in f.readlines():
            utt, spk = line.strip().split()
            utt2spk[utt] = spk
    return utt2spk


def feats_cut(feat, max_len):
    # basically concat data is actually not meaningful
    # to do : more elegant method to pad data
    if feat.shape[0] <= max_len:
        num_slice = max_len//feat.shape[0] + 1
        feats = [feat for i in range(num_slice)]
        feat = np.concatenate(feats, 0)
        feat = feat[:max_len]
    else:
        idx = random.randint(0, feat.shape[0] - max_len - 1)
        feat = feat[idx: idx + max_len]
    assert feat.shape[0] == max_len
    return feat


@DATASETS.register_module()
class SpeakerDataset(BaseDataset):
    def __init__(self, data_prefix, pipeline, map, do_chunk=False, chunksize=200):
        # here do_chunk means specify chunk position
        with open(map, "r") as f:
            self.map = json.load(f)
        self.do_chunk = do_chunk
        self.chunksize = chunksize
        self.spk2utt_path = os.path.join(data_prefix, "spk2utt")
        self.utt2spk_path = os.path.join(data_prefix, "utt2spk")
        self.scp_path = os.path.join(data_prefix, "feats.scp")
        self.spk2utt = get_spk2utt(self.spk2utt_path)
        self.utt2spk = get_utt2spk(self.utt2spk_path)
        super().__init__(data_prefix, pipeline, None, False)

    def load_annotations(self):
        data_infos = []
        with open(self.scp_path) as f:
            for line in f:
                utt, ark = line.strip().split()
                spk = self.utt2spk[utt]
                target = self.map[spk]
                data_infos.append({'img': ark, 'gt_label': target})
        return data_infos

    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        feat = kaldiio.load_mat(results["img"])
        feat = feats_cut(feat, self.chunksize)
        feat = feat.astype(np.float32)
        results["img"] = feat
        return self.pipeline(results)


if __name__ == "__main__":
    dataset = SpeakerDataset(
        "/data1/mayufeng/data_vox/train",
        [
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='ToTensor', keys=['feat'])
        ],
        "/data1/mayufeng/data_vox/train/uid2classes.json",
    )
    print(dataset[1000])