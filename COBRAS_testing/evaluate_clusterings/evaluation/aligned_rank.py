import collections
import numpy as np
import json
import os




def calculate_aligned_rank(scores, datasets):
    '''
    Scores maps number of constraints to dataset to method to score
    '''
    differences_to_mean = collections.defaultdict(list)
    for num_constraints in scores:
        for dataset in datasets:
            mean_score = np.mean(list(scores[num_constraints][dataset].values()))

            for algo in scores[num_constraints][dataset]:
                differences_to_mean[num_constraints].append(
                    (scores[num_constraints][dataset][algo] - mean_score, algo, dataset))

    ranks = collections.defaultdict(list)

    for nc in scores:
        sorted_diff = sorted(differences_to_mean[nc], key=lambda x: x[0], reverse=True)

        cur_aligned_ranks = collections.defaultdict(int)

        for diff_idx, (cur_diff, method, dataset) in enumerate(sorted_diff):
            cur_aligned_ranks[method] += diff_idx + 1

        for method in cur_aligned_ranks:
            cur_aligned_ranks[method] /= float(len(datasets))

        for method in cur_aligned_ranks:
            # ranks[method].append(cur_aligned_ranks[method] / len(datasets))
            ranks[method].append(cur_aligned_ranks[method])

    return ranks
