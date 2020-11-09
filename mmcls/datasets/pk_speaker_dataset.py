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
        spkr_dataset = {}
        for idx, item_dict in enumerate(self.data_infos):
            ark = item_dict["img"]
            target = item_dict["gt_label"]
            lst = spkr_dataset.get(target, [])
            lst.append([ark, idx])
            spkr_dataset[target] = lst
        return spkr_dataset