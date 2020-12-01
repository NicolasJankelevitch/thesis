import json
import math
import os
import matplotlib.pyplot as plt
import pylab
from matplotlib.pyplot import Axes
from config import FOLD_RESULT_DIR, FIGURE_DIR
from datasets import Dataset


def make_algos_dict_and_common_datasets(algo_names,query_budget):
    algos = dict()
    common_dataset_names = None
    for algo_name in algo_names:
        algo_path = os.path.join(FOLD_RESULT_DIR,algo_name,  f"runtime_averages_{query_budget}.txt")

        with open(algo_path, 'r') as file:
            runtime_dict = json.load(file)
        algos[algo_name] = runtime_dict
        if common_dataset_names is None:
            common_dataset_names = set(runtime_dict.keys())
        else:
            common_dataset_names = common_dataset_names.intersection(runtime_dict.keys())
    return algos, common_dataset_names

def plot_runtime_comparison(comparison_name, algo_names, line_names, colors, query_budget, legend = True):
    fig, ax = plt.subplots(figsize=(12, 8))

    ax: Axes = ax
    data_dict, common_datasets = make_algos_dict_and_common_datasets(algo_names, query_budget)
    common_datasets_size = [(Dataset(dataset_name).number_of_instances(), dataset_name) for dataset_name in common_datasets]
    reference = algo_names[0]
    for _, dataset_name in sorted(common_datasets_size):
        reference_time = data_dict[reference][dataset_name]
        for algo, algo_name, color in zip(algo_names, line_names, colors):
            time = data_dict[algo][dataset_name]
            relative_speedup = math.log10(time/reference_time)

            ax.scatter(dataset_name,relative_speedup,c=color, s = 70)
    output_file_name = os.path.join(FIGURE_DIR, comparison_name, f"average_runtimes_{query_budget}.png")
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
    ax.set_ylabel("speedup of "+line_names[0]+" over competitor")
    # ax.set_yscale('log')
    if legend:
        ax.legend(line_names, loc='upper left', bbox_to_anchor=(1,1))

    ax.set_yticks(range(-3,3))
    locs = ax.get_yticks()
    ax.set_yticklabels(10.0**loc if loc < 0 else int(10.0**loc) for loc in locs)
    ax.set_xticks([], [])
    ax.set_xlabel("datasets (sorted by size)")
    ax.set_title("query budget = {}".format(query_budget))
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(output_file_name, dpi=300, bbox_inches='tight')
    # figLegend = pylab.figure(figsize=(10, 10))
    # h, l = ax.get_legend_handles_labels()
    # pylab.figlegend(h, line_names, loc='upper left')
    # figLegend.savefig(os.path.join(os.path.dirname(output_file_name),"legend_{}.png".format(query_budget)))
    plt.close(fig)
