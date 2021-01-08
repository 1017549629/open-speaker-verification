import argparse
from kaldiio import ReadHelper
import tqdm
import numpy as np
from collections import defaultdict


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--whole_trial_embedding_path", type=str,
                        default="./xvector_train/vector.scp",
                        help="whole trial means including enroll and test in this scp file,"
                             "remember the utt-id in this scp should be unique")
    parser.add_argument("--cohort_embedding_path", type=str,
                        default="./xvector_cohort/vector.scp",
                        help="cohort embedding scp file path")
    parser.add_argument('--raw_scores', type=str, required=True, help="raw score file")
    parser.add_argument('--topN', tpye=int, default=400)
    parser.add_argument('--snorm', action='store_true',
                        help="decide do snorm or asnorm")
    parser.add_argument('--result_snorm', type=str, required=True)
    args = parser.parse_args()
    return args


def normalize(mat):
    mat_norm = np.linalg.norm(mat, axis=1).reshape([mat.shape[0], 1])
    mat = mat / mat_norm
    return mat


def generate_stat(whole_mat, cohort_mat, whole_idx_dict, do_asnorm, topN):
    inverse_whole_idx_dict = {v: k for k, v in whole_idx_dict.items()}
    stats_dict = {}

    "Since the whole matrix is way too big, we could slice our matrix"
    whole_mat = normalize(whole_mat)
    cohort_mat = normalize(cohort_mat)
    print("slicing enroll mat with 2048")
    step = whole_mat.shape[0]//2048 + 1
    print("steps:", step)
    for i in tqdm.tqdm(range(step)):
        start = i*2048
        end = min(i*2048+2048, whole_mat.shape[0])
        # start from "start" to "end" rows of enroll_mat
        scores = np.matmul(whole_mat[start: end, :], cohort_mat.T)
        if do_asnorm:
            # according to adaptive score normalization
            # topN highest scores are selected
            scores = np.sort(scores, axis=1)
            scores = scores[:, -topN:]
        # means and stds of scores are calculated
        means = np.mean(scores, axis=1)
        stds = np.std(scores, axis=1)
        for ind in range(start, end):
            key = inverse_whole_idx_dict[ind]
            stats_dict[key] = [means[ind-start], stds[ind-start]]
        return stats_dict


def output_score(raw_scores, stats_dict, output_file):
    result = open(output_file, "w")
    with open(raw_scores, "r") as scores:
        for line in tqdm.tqdm(scores):
            key1, key2, score = line.strip().split()
            score = float(score)
            mean1, std1 = stats_dict[key1]
            mean2, std2 = stats_dict[key2]

            # according to Symmetric Score Normalization, ZNORM adn TNORM scores are averaged
            znorm_score = (score - mean1) / std1
            tnorm_score = (score - mean2) / std2
            snorm_score = (znorm_score + tnorm_score) / 2
            result.write("%s %s %f\n" % (key1, key2, snorm_score))
    result.close()


def main():
    opt = parse_opt()

    whole_mat = []
    cohort_mat = []
    whole_idx_dict = defaultdict(lambda: len(whole_idx_dict))

    whole = 'scp:' + opt.whole_trial_embedding_path
    cohort = 'scp:' + opt.cohort_embedding_path
    with ReadHelper(whole) as reader:
        for key, numpy_array in reader:
            whole_mat.append(numpy_array)
            whole_idx_dict[key]
    with ReadHelper(cohort) as reader:
        for key, numpy_array in reader:
            cohort_mat.append(numpy_array)
    whole_mat = np.stack(whole_mat)
    cohort_mat = np.stack(cohort_mat)

    assert whole_mat.ndim == 2 and cohort_mat.ndim == 2, "dimension should be 2"
    do_asnorm = not opt.snorm
    stats_dict = generate_stat(whole_mat, cohort_mat, whole_idx_dict, do_asnorm, opt.topN)
    output_score(opt.raw_scores, stats_dict, opt.result_snorm)


if __name__ == '__main__':
    main()

