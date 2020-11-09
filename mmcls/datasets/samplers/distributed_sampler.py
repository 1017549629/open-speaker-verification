import torch
from torch.utils.data import DistributedSampler as _DistributedSampler
# from mmcls.datasets import PKSpeakerDataset


class DistributedSampler(_DistributedSampler):

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 round_up=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle
        self.round_up = round_up
        if self.round_up:
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(self.dataset)

        # added to adapt PK sampling strategy
        self.do_pk = hasattr(dataset, "K")
        if self.do_pk:
            print("Start using PK sampling strategy!")
            self.spkr_dataset_ids = dataset.spkr_dataset_ids
            self.K = dataset.K
            self.P = dataset.P

    def __iter__(self):
        if not self.do_pk:
            # deterministically shuffle based on epoch
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.epoch)
                indices = torch.randperm(len(self.dataset), generator=g).tolist()
            else:
                indices = torch.arange(len(self.dataset)).tolist()

            # add extra samples to make it evenly divisible
            if self.round_up:
                indices += indices[:(self.total_size - len(indices))]
            assert len(indices) == self.total_size

            # subsample
            indices = indices[self.rank:self.total_size:self.num_replicas]
            if self.round_up:
                assert len(indices) == self.num_samples

            return iter(indices)
        else:
            lol = lambda lst, sz: [lst[i:i + sz] for i in range(0, len(lst), sz)]
            items = list(self.spkr_dataset_ids.items())

            # metric learning naturally needs shuffle to be True
            g = torch.Generator()
            g.manual_seed(self.epoch)

            flattened_list = []
            flattened_label = []

            for spkr, ark_idx in items:
                ids = [d[1] for d in ark_idx]
                numSeg = (len(ids) // self.K) * self.K
                rp = lol(torch.randperm(len(ids), generator=g).tolist()[:numSeg], self.K)
                flattened_label.extend([spkr]*len(rp))
                for indices in rp:
                    flattened_list.append([ids[i] for i in indices])

            mixid = torch.randperm(len(flattened_label), generator=g).tolist()
            mixlabel = []
            mixmap = []

            assert self.batch_size % self.K, \
                "batchsize %d should be exactly divided by K %d" % (self.batch_size, self.K)
            tuple_batch_size = self.batch_size // self.K

            for ii in mixid:
                startbatch = len(mixlabel) - len(mixlabel) % tuple_batch_size
                if flattened_label[ii] not in mixlabel[startbatch:]:
                    mixlabel.append(flattened_label[ii])
                    mixmap.append(ii)

            all_indices = []
            for idx in mixmap:
                all_indices.extend(flattened_list[idx])
            round_len = len(all_indices) // (self.num_replicas * self.batch_size) * self.batch_size
            sub_indices = all_indices[self.rank * round_len : (self.rank+1) * round_len]
            return iter(sub_indices)
