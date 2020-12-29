#### added by Yufeng
# This script tends to transform the kaldi format to h5 or csv format which can be read through vaex.


import argparse
from collections import defaultdict
import os
from tqdm import tqdm
import vaex


parser = argparse.ArgumentParser()
parser.add_argument("--feat_scp", type=str, required=True,
                    help="feats.scp file in kaldi format")
parser.add_argument("--utt2spk", type=str, required=True, help="utt2spk in kaldi format")
parser.add_argument("--out_dir", type=str, required=True, help="output directory")
parser.add_argument("--data_type", type=str, default="train")
args = parser.parse_args()


def get_utt2spk(file):
    utt2spk = {}
    spk_dict = defaultdict(lambda :len(spk_dict))
    with open(file, "r") as f:
        for line in tqdm(f.readlines()):
            utt, spk = line.strip().split()
            utt2spk[utt] = spk
            # do nothing in lambda dict
            spk_dict[spk]
    return utt2spk, spk_dict


def feats2csv(feat_scp, out_dir, utt2spk, spk_dict):
    os.makedirs(out_dir, exist_ok=True)
    feat_name = feat_scp.rstrip(".scp").split("/")[-1]
    # print(feat_name)
    feats = open(feat_scp, "r")
    out_csv = os.path.join(out_dir, feat_name + "." + args.data_type + ".csv")
    with open(out_csv, "w") as f:
        f.write("utt_id ark_path class_label\n")
        for feat in feats:
            utt_id, ark_path = feat.rstrip().split()
            class_label = spk_dict[utt2spk[utt_id]]
            f.write("%s %s %d\n" % (utt_id, ark_path, class_label))
    feats.close()

    # convert csv to h5 file
    df = vaex.from_csv(out_csv, sep=" ")
    df = df.sort(by="class_label")
    h5_name = os.path.join(out_dir, feat_name + "." + args.data_type + ".hdf5")
    df.export(h5_name)
    df.export(out_csv)


def main():
    utt2spk, spk_dict = get_utt2spk(args.utt2spk)
    feats2csv(args.feat_scp, args.out_dir, utt2spk, spk_dict)


if __name__ == "__main__":
    main()