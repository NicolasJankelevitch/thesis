import collections
import os
import json
from config import FOLD_RESULT_DIR,FIGURE_DIR
from evaluate_clusterings.evaluation.aligned_rank import calculate_aligned_rank


def read_average_aris(algo_name):
    with open(os.path.join(FOLD_RESULT_DIR,algo_name, "ari_averages.txt"), mode = 'r') as file:
        averages_dict = json.load(file)
    return averages_dict


def calculate_and_write_aligned_rank(algo_names, comparison_name, dataset_names = None):
    print("calculating average rank")
    # maps number of constraints to a dataset to a method to a score
    scores = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(float)))
    algorithm_dict = dict()
    for algo_name in algo_names:
        averages_dict = read_average_aris(algo_name)
        algorithm_dict[algo_name] = averages_dict
    common_dataset_names = dataset_names
    for algo_name in algo_names:
        averages_dict = algorithm_dict[algo_name]
        if common_dataset_names is None:
            common_dataset_names = set(averages_dict.keys())
        else:
            common_dataset_names = common_dataset_names.intersection(averages_dict.keys())
    common_dataset_names.remove("overall_average")

    for algo_name in algo_names:
        averages_dict = algorithm_dict[algo_name]
        for dataset in averages_dict:
            if dataset not in common_dataset_names:
                continue
            average_ARI = averages_dict[dataset]
            # if limitsize is not None:
            #     average_ARI = average_ARI[:limitsize]
            for i in range(len(average_ARI)):
                scores[i][dataset][algo_name] = average_ARI[i]

    ranks = calculate_aligned_rank(scores, common_dataset_names)
    result_file = os.path.join(FIGURE_DIR, comparison_name, "rank.txt")
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, mode = 'w') as rank_file:
        json.dump(ranks, rank_file)
