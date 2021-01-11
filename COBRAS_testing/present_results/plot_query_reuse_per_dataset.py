from config import FOLD_RESULT_DIR
import os
import matplotlib.pyplot as plt
from config import FIGURE_DIR
import numpy as np
import ast


def calculate_and_plot_query_reuse_per_dataset(comparison_name, test_names):
    dataset_names = os.listdir(os.path.join(os.path.join(FOLD_RESULT_DIR, test_names[0]), "clusterings"))

    for dataset_name in dataset_names:
        reuse_list = []
        unique_reuse_list = []
        unique_list = []
        for test_name in test_names:
            path = os.path.join(os.path.join(os.path.join(FOLD_RESULT_DIR, test_name), "clusterings"), dataset_name)
            reused_test = 0
            unique_test = 0
            unique_reuse_test = 0
            for f in os.listdir(path):
                file = os.path.join(path, f)
                with open(file) as d:
                    data = d.read()
                    dictionary = ast.literal_eval(data)
                    reused_file = dictionary['reused_constraints']
                    unique_file = len(dictionary['mls']) + len(dictionary['cls']) + len(dictionary['dks'])
                    unique_reuse_file = get_unique_set(reused_file)
                reused_test = reused_test + len(reused_file)
                unique_test = unique_test + unique_file
                unique_reuse_test = unique_reuse_test + len(unique_reuse_file)
            reuse_list.append(reused_test)
            unique_reuse_list.append(unique_reuse_test)
            unique_list.append(unique_test)
        image_loc = os.path.join(os.path.join(FIGURE_DIR, "reuse_per_dataset"), dataset_name+".png")
        plot(dataset_name, image_loc, test_names, reuse_list, unique_reuse_list, unique_list)

def plot(comparison_name, image_loc, test_names, reuse_list, unique_reused_list, unique_list):
    x = np.arange(len(test_names))
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(x + 0.00, reuse_list, label="#constraints reused", width=0.25)
    ax.bar(x + 0.25, unique_reused_list, label="#unique reused constraints", width=0.25)
    ax.bar(x - 0.25, unique_list, label="#unique constraints", width=0.25)
    ax.set_xticks(x)
    ax.set_xticklabels(test_names)
    plt.xticks(rotation=90)
    ax.legend()
    output_file_name = image_loc
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
    plt.savefig(output_file_name, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("plotted {}".format(image_loc))

def get_unique_set(constraints):
    return_set = set()
    for val in constraints:
        list = next(iter(val.values()))
        return_set.add((list[0], list[1]))
    return return_set


if __name__ == '__main__':
    calculate_and_plot_query_reuse_per_dataset("bar_graph_per_dataset", ["standard_COBRAS", "6", "9", "10"])
