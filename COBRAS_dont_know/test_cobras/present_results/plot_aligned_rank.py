import json
from util.config import FIGURES_PATH
import matplotlib.pyplot as plt
import os


def plot_rank_comparison_file(comparison_name, algo_names, line_names):
    with open(os.path.join(FIGURES_PATH, comparison_name,"rank.txt"), 'r') as file:
        result_dict = json.load(file)

    fig = plt.figure(figsize=(6,6))
    for idx, (algo,line_name) in enumerate(zip(algo_names,line_names)):
        line_color = "C{}".format(idx + 1)
        plt.plot(result_dict[algo], label=line_name, color=line_color)

    plt.ylabel("average aligned rank")
    plt.xlabel("number of queries")

    plt.legend()
    output_file_name=os.path.join(FIGURES_PATH, comparison_name, "aligned_rank.png")
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
    plt.savefig(output_file_name, dpi=300, bbox_inches='tight')

    plt.close(fig)
