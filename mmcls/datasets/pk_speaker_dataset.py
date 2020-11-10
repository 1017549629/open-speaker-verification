from .builder import DATASETS
from .speaker_dataset import SpeakerDataset


@DATASETS.register_module()
class PKSpeakerDataset(SpeakerDataset):
    """
    PKSpeakerDataset
    """

    def __init__(self, data_prefix, pipeline, map, P, K, do_chunk=False, chunksize=200):
        self.P = P
        self.K = K
        super(PKSpeakerDataset, self).__init__(data_prefix, pipeline, map, do_chunk, chunksize)
        self.spkr_dataset_ids = self.create_spkr_dataset()

    def create_spkr_dataset(self):
        spk2id = {}
        for i, dic in enumerate(self.data_infos):
            spk = dic["gt_label"]
            lst = spk2id.get(spk, [])
            lst.append(i)
            spk2id[spk] = lst
        return spk2id