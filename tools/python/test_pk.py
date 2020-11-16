from argparse import ArgumentParser
import numpy as np

np.random.seed(1)

parser = ArgumentParser("pk parser")
parser.add_argument("-f", dest="feats", required=True)
args = parser.parse_args()


def read_data(feats):
    spk2utt = {}
    with open(feats) as f:
        lines = f.readlines()
    print(len(lines))
    lines = [line.strip().split() for line in lines]
    for i, (key, _) in enumerate(lines):
        spk = key.split("-")[0]
        lst = spk2utt.get(spk, [])
        lst.append(i)
        spk2utt[spk] = lst
    return spk2utt


def sample(spk2ind, batchsize=128, K=8):
    lol = lambda lst, sz: [lst[i:i + sz] for i in range(0, len(lst), sz)]
    items = list(spk2ind.items())

    flattened_list = []
    flattened_label = []

    for spkr, indLst in items:
        numSeg = len(indLst)//K * K
        rp = lol(np.random.permutation(len(indLst)).tolist()[:numSeg], K)
        flattened_label.extend([spkr] * len(rp))
        for indices in rp:
            flattened_list.append([indLst[i] for i in indices])

    mixid = np.random.permutation(len(flattened_label)).tolist()
    mixlabel = []
    mixmap = []

    tuple_batch_size = batchsize // K
    for ii in mixid:
        startbatch = len(mixlabel) - len(mixlabel) % tuple_batch_size
        if flattened_label[ii] not in mixlabel[startbatch:]:
            mixlabel.append(flattened_label[ii])
            mixmap.append(ii)

    all_indices = []
    for idx in mixmap:
        all_indices.extend(flattened_list[idx])
    return all_indices


def main():
    spk2utt = read_data(args.feats)
    sampler = sample(spk2utt, batchsize=96, K=8)
    # print(sampler[:128])
    print(len(sampler))

main()