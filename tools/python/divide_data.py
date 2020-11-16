from argparse import ArgumentParser
import os
import tqdm
import random
import json

parser = ArgumentParser()
parser.add_argument("--dir", type=str, required=True)
parser.add_argument("--target", type=str, required=True)
parser.add_argument("--val_num", type=int, default=1)
parser.add_argument("--aug_types", nargs="+", default="music babble noise reverb")
args = parser.parse_args()


def read_pair(pfile):
    with open(pfile) as f:
        lines = f.readlines()
    return [line.strip().split() for line in tqdm.tqdm(lines)]


def read_prep(datadir):
    feats_path = os.path.join(datadir, "feats.scp")
    spk2utt_path = os.path.join(datadir, "spk2utt")
    utt2spk_path = os.path.join(datadir, "utt2spk")
    feats = read_pair(feats_path)
    utt2spk = {k: v for k, v in read_pair(utt2spk_path)}
    spk2utt = {k[0]: k[1:] for k in read_pair(spk2utt_path)}
    return feats, utt2spk, spk2utt


def divide_aug_ori(spk2utt, aug_types):
    spk_ori = {}
    spk_aug = {}
    for spk in spk2utt.keys():
        utts = spk2utt[spk]
        ori = []
        augs = []
        for utt in utts:
            is_ori = True
            for atype in aug_types:
                if utt.endswith(atype):
                    augs.append(utt)
                    is_ori = False
                    break
            if is_ori:
                ori.append(utt)
        assert len(ori) + len(augs) == len(utts)
        spk_ori[spk] = ori
        spk_aug[spk] = augs
    return spk_ori, spk_aug


def sample(spk_lst_ori, spk_lst_aug, val_num):
    train_lst = {}
    val_lst = {}
    for spk in tqdm.tqdm(spk_lst_ori.keys()):
        utts = spk_lst_ori[spk]
        random.shuffle(utts)
        val_utts = utts[:val_num]
        train_utts = utts[val_num:]
        train_lst[spk] = train_utts
        val_lst[spk] = val_utts
        utts = spk_lst_aug[spk]
        random.shuffle(utts)
        val_utts = utts[:val_num]
        train_utts = utts[val_num:]
        train_lst[spk].extend(train_utts)
        val_lst[spk].extend(val_utts)
    return train_lst, val_lst


def output(spk2utt, feats_scp, target):
    os.makedirs(target, exist_ok=True)
    feats_scp = {k:v for k,v in feats_scp}
    output_feats = open(os.path.join(target, "feats.scp"), "w")
    output_utt2spk = open(os.path.join(target, "utt2spk"), "w")
    output_spk2utt = open(os.path.join(target, "spk2utt"), "w")
    for spk in tqdm.tqdm(spk2utt.keys()):
        utts = spk2utt[spk]
        valid_utts = []
        for utt in utts:
            if utt in feats_scp.keys():
                valid_utts.append(utt)
                output_feats.write("%s %s\n" % (utt, feats_scp[utt]))
                output_utt2spk.write("%s %s\n" % (utt, spk))
        output_spk2utt.write("%s %s\n" % (spk, " ".join(valid_utts)))
    output_feats.close()
    output_spk2utt.close()
    output_utt2spk.close()


def create_map(dict_keys, target):
    d = {}
    for i, k in enumerate(list(dict_keys)):
        d[k] = i
    with open(os.path.join(target, "uid2classes.json"), "w") as f:
        json.dump(d, f)


def main():
    print("preparing feats")
    feats, utt2spk, spk2utt = read_prep(args.dir)
    spk_ori, spk_aug = divide_aug_ori(spk2utt, aug_types=args.aug_types)
    train_spk2utt, val_spk2utt = sample(spk_ori, spk_aug, args.val_num)
    output(train_spk2utt, feats, os.path.join(args.target, "train"))
    output(val_spk2utt, feats, os.path.join(args.target, "val"))
    create_map(spk2utt.keys(), os.path.join(args.target, "train"))


main()
