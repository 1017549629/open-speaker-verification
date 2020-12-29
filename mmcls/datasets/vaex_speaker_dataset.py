# created by Yufeng Ma
# email: wyrmyf@gmail.com

import kaldiio
import vaex
import copy
import numpy as np

from .builder import DATASETS
from .base_dataset import BaseDataset
from .speaker_dataset import feats_cut
from multiprocessing import Array
import random
import math


@DATASETS.register_module()
class VaexSpeakerDataset(BaseDataset):
    def __init__(self, data_prefix, pipeline, chunk_type="random", chunksize=200):
        self.chunksize = chunksize
        # chunk type will include random and pseudo and fix
        self.chunk_type = chunk_type
        super().__init__(data_prefix, pipeline, None, False)
        if self.chunk_type == "pseudo":
            # since pseudo, will create a shared_memory type Array
            self.positions = Array("i", [0 for ind in range(len(self.data_infos))])
            self.iters = Array("i", [0 for ind in range(len(self.data_infos))])

    def load_annotations(self):
        # self.data_infos right now is a vaex dataframe
        return vaex.open(self.data_prefix)

    def get_pseudo_random_feats(self, feat, idx):
        cur_iter = self.iters[idx]
        cur_pos = self.positions[idx]
        feat_len = feat.shape[0]
        iter_cycle = math.ceil(feat_len / self.chunksize)
        if cur_iter % iter_cycle == 0:
            # when the whole range of data are selected, start a new position
            cur_pos = random.randint(0, feat_len)

        cur_pos = cur_pos % feat_len

        if cur_pos + self.chunksize > feat_len:
            cur_pos = feat_len - self.chunksize

        self.positions[idx] = cur_pos + self.chunksize
        self.iters[idx] += 1
        return feat[cur_pos:cur_pos+self.chunksize, :]

    def prepare_data(self, idx):
        utt_id, ark_path, gt_label = copy.deepcopy(self.data_infos[idx])
        feat = kaldiio.load_mat(ark_path)
        feat = feat.astype(np.float32)
        if self.chunk_type == "random":
            feat = feats_cut(feat, self.chunksize)
        elif self.chunk_type == "fix":
            feat = feat[:self.chunksize, :]
        elif self.chunk_type == "pseudo":
            feat = self.get_pseudo_random_feats(feat, idx)
        feat = feat.astype(np.float32)
        results = {"img": feat, "gt_label": int(gt_label)}
        return self.pipeline(results)


if __name__ == "__main__":
    dataset = VaexSpeakerDataset(
        "/home/mayufeng/test_csv/feat.val.arrow",
        [
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='ToTensor', keys=['img'])
        ],
        "pseudo"
    )
    from torch.utils.data import DataLoader

    for i in range(4):
        loader = DataLoader(dataset, batch_size=128, num_workers=4)
        for batch_ndx, sample in enumerate(loader):
            pass
        print(dataset.positions[10], dataset.iters[10])