import argparse
import ast
import math
import kaldiio
import numpy as np
from multiprocessing import Pool
from kaldiio import WriteHelper
import os
import torch
from glob import glob
import collections
from tqdm import tqdm
import mmcv
from mmcls.models import build_classifier
from mmcv.runner import load_checkpoint
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--is_vector_average", type=int, default=0)
    parser.add_argument("--feats_scp", type=str, default="")
    parser.add_argument("--utt2spk", type=str, default="")
    parser.add_argument("--model_file", type=str, default="")
    parser.add_argument("--save_vector_path", type=str, default="")
    parser.add_argument("--vector_scp", type=str, default="vector.scp")

    parser.add_argument("--min_len", type=int, default=100, help="")
    parser.add_argument("--gpus", type=str, default="1, 2, 3", help="")
    parser.add_argument("--num_process", type=int, default=4, help="")

    parser.add_argument("--do_fixed_length", type=bool, default=True)
    parser.add_argument("--fixed_length", type=int, default=60000)
    parser.add_argument("--keep_res", type=bool, default=True)

    args = parser.parse_args()
    return args


def div_process(file, num_process):
    utt_list = []
    f = open(file)
    for line in f.readlines():
        line = line.strip()
        utt, ark = line.split()[0], line.split()[1]
        utt_list.append([utt, ark])
    each_num = math.ceil(len(utt_list) / num_process)
    utt_process = [utt_list[i:i + each_num] for i in range(0, len(utt_list), each_num)]
    return utt_process


def fix_data(feat, fixed_length):
    le = feat.shape[0]
    if le > fixed_length:
        num_slices = le//fixed_length+1
        new_feat = [feat[i*fixed_length:((i+1)*fixed_length)] for i in range(num_slices-1)]
        if le % fixed_length != 0:
            new_feat.append(feat[-(le % fixed_length):])
    else:
        new_feat = [feat]
    return new_feat


class Extract_dataset():
    def __init__(self, samples):
        self.samples = samples

    def __getitem__(self, index):
        key, rxfile = self.samples[index]
        feat = kaldiio.load_mat(rxfile)
        feat = feat.astype(np.float32)
        return feat, key

    def __len__(self):
        return len(self.samples)


def try_extract_feature(checkpoint, feats, process_id, cfg):
    try:
        extract_feature(checkpoint, feats, process_id, cfg)
    except Exception as e:
        print(e)
        raise e


def load_model(checkpoint, config):
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    model = build_classifier(config.model)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)
    model.cfg = config  # save the config in the model for convenience
    model.cuda()
    model.eval()
    return model


def extract_feature(checkpoint, feats, process_id, cfg):
    if cfg.gpus == "-1":
        device="cpu"
    else:
        device = "cuda"
        gpus = [int(i) for i in cfg.gpus.split(",")]
        num_gpus = len(gpus)
        current_gpu = str(gpus[int(process_id // (cfg.num_process / num_gpus))])
        os.environ["CUDA_VISIBLE_DEVICES"] = current_gpu
        if torch.cuda.is_available():
            print("current gpu is {}".format(current_gpu))
        else:
            assert Exception("current gpu {} is not available".format(current_gpu))

    model = load_model(checkpoint, cfg.config)
    model.to(device)
    data_loader = Extract_dataset(feats)

    save_vector_path = os.path.join(cfg.save_vector_path, "vector_save")
    os.makedirs(save_vector_path, exist_ok=True)
    save_name = os.path.join(save_vector_path, "vector.{}".format(str(process_id)))
    scp_save = "ark,scp:{}.ark,{}.scp".format(save_name, save_name)

    with WriteHelper(scp_save) as writer:
        with torch.no_grad():
            for i, (data, utt) in tqdm(enumerate(data_loader)):
                if cfg.do_fixed_length:
                    # here data is a list which is not a concatenated batch
                    data = fix_data(data, cfg.fixed_length)
                else:
                    data = np.expand_dims(data, 0)
                vectors = []
                lens = []
                for slice in data:
                    slice = torch.from_numpy(slice)
                    slice.unsqueeze_(0)
                    le = slice.shape[1]
                    if le < cfg.min_len:
                        if cfg.keep_res:
                            left_context = int((cfg.min_len - le) / 2)
                            right_context = cfg.min_len - le - left_context
#                            padding = torch.zeros(1, cfg.min_len-le, slice.shape[2])
                            left_padding = torch.ones(1, left_context, slice.shape[2]) * slice[0][0]
                            right_padding = torch.ones(1, right_context, slice.shape[2]) * slice[0][le - 1]
                            slice = torch.cat([left_padding, slice, right_padding], dim=1)
                            print("---padding---")
                        else:
                            continue
                    lens.append(le)
                    slice = slice.to(device)
                    feature = model.extract_feat(slice)
                    vectors.append(feature.cpu().numpy())

                averaged_feature = np.array([vectors[i] * lens[i] for i in range(len(vectors))]).sum(axis=0)/sum(lens)
                averaged_feature = averaged_feature.squeeze(0)
                writer(utt, averaged_feature)
                # print(utt)


def multiprocess_extract(feats_process, cfg):
    pool = Pool(cfg.num_process)
    retList = []
    for process_id, feats_list in enumerate(feats_process):
        ret = pool.apply_async(try_extract_feature, args=(cfg.model_file, feats_list, process_id, cfg))
        retList.append(ret)
    for ret in retList:
        ret.get()
    pool.close()
    pool.join()
    scp_list = glob(os.path.join(cfg.save_vector_path, "vector_save", "vector.*.scp"))
    content = []
    for scp in scp_list:
        with open(scp, 'rb') as f:
            content = content + f.readlines()
    xvector_scp = os.path.join(cfg.save_vector_path, cfg.vector_scp)
    with open(xvector_scp, 'wb') as f:
        f.writelines(content)

    return xvector_scp


def generate_list_file(feats_path, data_utt2spk, save_data_path, is_vector_average=False):
    if os.path.exists(data_utt2spk):
        data_utt2spk_dict = {}
        with open(data_utt2spk, "r") as f:
            for line in f.readlines():
                utt, spk = line.strip().split()
                data_utt2spk_dict[utt] = spk

    spk2utt = collections.defaultdict(list)
    with open(os.path.join(save_data_path, "utt2spk"), "w") as f_utt2spk:
        with open(feats_path, "r") as f_feats:
            for line in f_feats.readlines():
                line = line.strip()
                utt = line.split(" ")[0]
                if is_vector_average:
                    spk = utt
                elif os.path.exists(data_utt2spk):
                    spk = data_utt2spk_dict[utt]
                else:
                    spk = utt.split("-")[0]
                f_utt2spk.write(utt + " " + spk + "\n")
                spk2utt[spk].append(utt)

    with open(os.path.join(save_data_path, "spk2utt"), "w") as f_spk2utt:
        with open(os.path.join(save_data_path, "spk2num"), "w") as f_spk2num:
            for i, (spk, utt_list) in enumerate(spk2utt.items()):
                f_spk2utt.write(spk + " " + " ".join(utt_list) + "\n")
                f_spk2num.write(spk + " " + str(len(utt_list)) + "\n")


def vector_average(xvector_scp, utt2spk):
    if os.path.exists(utt2spk):
        data_utt2spk_dict = {}
        with open(utt2spk, "r") as f:
            for line in f.readlines():
                utt, spk = line.strip().split()
                data_utt2spk_dict[utt] = spk

    spk2vector = collections.defaultdict(list)
    with open(xvector_scp) as f:
        for line in f.readlines():
            line = line.strip()
            utt, ark = line.split()[0], line.split()[1]
            vector = kaldiio.load_mat(ark)
            if os.path.exists(utt2spk):
                spk2vector[data_utt2spk_dict[utt]].append(vector)
            else:
                spk = utt.split("-")[0]
                spk2vector[spk].append(vector)

    xvector_average = xvector_scp.split(".scp")[0]
    scp_save = "ark,scp:{}.ark,{}.scp".format(xvector_average, xvector_average)
    with WriteHelper(scp_save) as writer:
        for spk, vector_list in spk2vector.items():
            feats = np.mean(np.array(vector_list), axis=0)
            writer(spk, feats)


def main():
    args = parse_args()
    
    feats_process = div_process(args.feats_scp, args.num_process)
    xvector_scp = multiprocess_extract(feats_process, args)
    generate_list_file(xvector_scp, args.utt2spk, args.save_vector_path)
    if args.is_vector_average:
        print("vector average ......")
        vector_average(xvector_scp, args.utt2spk)
        generate_list_file(xvector_scp, args.utt2spk, args.save_vector_path, is_vector_average=args.is_vector_average)


if __name__ == "__main__":
    main()
