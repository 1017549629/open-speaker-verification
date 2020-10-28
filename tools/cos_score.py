import argparse
from kaldiio import ReadHelper
import tqdm
import numpy as np
from collections import defaultdict


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_embedding_path", type=str, default="./xvector_test/vector.scp")
    parser.add_argument("--enroll_embedding_path", type=str, default="./xvector_train/vector.scp")
    parser.add_argument("--is_mean", type=int, default=0)
    parser.add_argument("--mean_vec", type=str, default="self")
    parser.add_argument("--train_embedding_path", type=str, default="")
    parser.add_argument('--trial_list', type=str, default="")
    parser.add_argument('--result_cosine', type=str, default='')
    args = parser.parse_args()
    return args


def normalize(mat):
    mat_norm = np.linalg.norm(mat, axis=1).reshape([mat.shape[0], 1])
    mat = mat / mat_norm
    return mat


def generate_score(enroll_mat, test_mat, enroll_idx_dict,
                   test_idx_dict, test_pair, result_path):
    print("test_pair: ", test_pair)
    output_dict = {}
    with open(test_pair, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in tqdm.tqdm(lines):
            new_line = line.strip().split(" ")
            if len(new_line) == 2:
                key1, key2 = new_line
            else:
                key1, key2, label = new_line
            if key1 in output_dict:
                output_dict[key1].append(key2)
            else:
                output_dict[key1] = [key2]

    inverse_enroll_idx_dict = {v:k for k,v in enroll_idx_dict.items()}
    """calculate EER"""
    "Since the whole matrix is way too big num_enroll*num_test, we could slice our matrix"
    enroll_mat = normalize(enroll_mat)
    test_mat = normalize(test_mat)
    print("slicing enroll mat with 2048")
    step = enroll_mat.shape[0]//2048 + 1
    f = open(result_path, "w", encoding="utf-8")
    print("steps:", step)
    for i in tqdm.tqdm(range(step)):
        start = i*2048
        end = min(i*2048+2048, enroll_mat.shape[0])
        # start from "start" to "end" rows of enroll_mat
        scores = np.matmul(enroll_mat[start: end, :], test_mat.T)
        for ind in range(start, end):
            query = inverse_enroll_idx_dict[ind]
            a_list = output_dict[query]
            for a_key in a_list:
                a_id = test_idx_dict[a_key]
                distance = scores[ind-start, a_id]
                f.write(query + " " + a_key + " " + str(distance) + "\n")
    f.close()


def centering_mean(enroll_mat):
    center_feat = np.mean(enroll_mat, axis=0)
    return center_feat


def main():
    opt = parse_opt()

    enroll_mat = []
    test_mat = []
    enroll_idx_dict = defaultdict(lambda: len(enroll_idx_dict))
    test_idx_dict = defaultdict(lambda: len(test_idx_dict))

    enroll = 'scp:' + opt.enroll_embedding_path
    verify = 'scp:' + opt.test_embedding_path
    with ReadHelper(enroll) as reader:
        for key, numpy_array in reader:
            enroll_mat.append(numpy_array)
            enroll_idx_dict[key]
    with ReadHelper(verify) as reader:
        for key, numpy_array in reader:
            test_mat.append(numpy_array)
            test_idx_dict[key]
    enroll_mat = np.stack(enroll_mat)
    test_mat = np.stack(test_mat)
    # print(enroll_mat.shape)
    # print(test_mat.shape)

    if opt.is_mean:
        if opt.mean_vec == "self":
            mean_feat = centering_mean(enroll_mat)
        else:
            mean_feat = np.loadtxt(opt.mean_vec)
        enroll_mat = enroll_mat - mean_feat
        test_mat = enroll_mat - mean_feat

    enroll_idx_dict = dict(enroll_idx_dict)
    test_idx_dict = dict(test_idx_dict)
    assert enroll_mat.ndim == 2, "dimension should be 2"
    generate_score(enroll_mat, test_mat, enroll_idx_dict, test_idx_dict,
                   opt.trial_list, opt.result_cosine)


if __name__ == '__main__':
    main()

