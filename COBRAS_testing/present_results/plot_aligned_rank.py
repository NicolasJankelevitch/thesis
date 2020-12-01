import json


from config import FIGURE_DIR
import matplotlib.pyplot as plt
import os

def plot_rank_comparison_file(comparison_name, algo_names, line_names):
    with open(os.path.join(FIGURE_DIR, comparison_name,"rank.txt"), 'r') as file:
        result_dict = json.load(file)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for idx, (algo,line_name) in enumerate(zip(algo_names,line_names)):
        ofset = 0
        scores = result_dict[algo]
        line_color = "C{}".format(idx + 1)
        plt.plot(scores, label = line_name, color=line_color)
        plt.text(len(scores), scores[-1] + ofset, line_name, c=line_color, horizontalalignment='left',
                 verticalalignment='center')

    plt.ylabel("average aligned rank")
    plt.xlabel("number of queries")

    # plt.legend()
    output_file_name =os.path.join(FIGURE_DIR, comparison_name, "aligned_rank.png")
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
    plt.savefig(output_file_name,dpi=300, bbox_inches = 'tight')

    plt.close(fig)
