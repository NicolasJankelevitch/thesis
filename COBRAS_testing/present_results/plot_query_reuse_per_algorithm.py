from config import FOLD_RESULT_DIR
import os
import matplotlib.pyplot as plt
from config import FIGURE_DIR
import numpy as np
import ast


def calculate_and_plot_query_reuse_per_algorithm(comparison_name, test_names):
    for test_name in test_names:
        test_path = os.path.join(os.path.join(FOLD_RESULT_DIR, test_name), "clusterings")
        dataset_names = os.listdir(test_path)
        reuse_list = []
        unique_reuse_list = []
        unique_list = []
        for dataset_name in dataset_names:
            dataset_folder = os.path.join(test_path, dataset_name)
            total_dataset_reuses = 0
            total_dataset_unique_reuse = 0
            total_dataset_unique = 0
            for f in os.listdir(dataset_folder):
                file = os.path.join(dataset_folder, f)
                with open(file) as d:
                    data = d.read()
                    dictionary = ast.literal_eval(data)
                    reused_file = dictionary['reused_constraints']
                    unique_file = len(dictionary['mls']) + len(dictionary['cls']) + len(dictionary['dks'])
                    unique_reuse_file = get_unique_set(reused_file)
                total_dataset_reuses = total_dataset_reuses + len(reused_file)
                total_dataset_unique_reuse = total_dataset_unique_reuse + len(unique_reuse_file)
                total_dataset_unique = total_dataset_unique + unique_file
            reuse_list.append(total_dataset_reuses)
            unique_reuse_list.append(total_dataset_unique_reuse)
            unique_list.append(total_dataset_unique)
        image_name = comparison_name + "_" + test_name + ".png"
        plot(comparison_name, image_name, dataset_names, reuse_list, unique_reuse_list, unique_list)


def plot(comparison_name, image_name, test_names, reuse_list, unique_reused_list, unique_list):
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
    output_file_name = os.path.join(FIGURE_DIR, comparison_name, image_name)
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
    plt.savefig(output_file_name, dpi=300, bbox_inches='tight')
    print("plotted {}".format(image_name))


def get_unique_set(constraints):
    return_set = set()
    for val in constraints:
        list = next(iter(val.values()))
        return_set.add((list[0], list[1]))
    return return_set


if __name__ == '__main__':
    calculate_and_plot_query_reuse_per_algorithm("bar_graph_per_dataset", ["standard_COBRAS", "6", "9", "10"])
