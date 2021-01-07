from config import FOLD_RESULT_DIR
import os
import matplotlib.pyplot as plt
from config import FIGURE_DIR
import numpy as np
import ast


def calculate_and_plot_query_reuse(comparison_name, test_names):
    reuse_list = []
    unique_reused_list = []
    unique_list = []

    for test_name in test_names:
        test_path = os.path.join(FOLD_RESULT_DIR, test_name)
        test_path = os.path.join(test_path, "clusterings")
        test_reuses = 0
        test_uniques_reused = 0
        test_unique = 0

        for dataset_name in os.listdir(test_path):
            dataset_folder = os.path.join(test_path, dataset_name)
            total_dataset_reuses = 0
            total_dataset_unique_reused = 0
            total_dataset_unique = 0
            for f in os.listdir(dataset_folder):
                file = os.path.join(dataset_folder, f)
                with open(file) as d:
                    data = d.read()
                    dictionary = ast.literal_eval(data)
                    reused_constraints = dictionary['reused_constraints']
                    unique_constraints = len(dictionary['mls']) + len(dictionary['cls']) + len(dictionary['dks'])
                    unique_reused_constraints = get_unique_set(reused_constraints)
                total_dataset_reuses = total_dataset_reuses + len(reused_constraints)
                total_dataset_unique_reused = total_dataset_unique_reused + len(unique_reused_constraints)
                total_dataset_unique = total_dataset_unique + unique_constraints

            test_reuses = test_reuses + total_dataset_reuses
            test_uniques_reused = test_uniques_reused + total_dataset_unique_reused
            test_unique = test_unique + total_dataset_unique

        reuse_list.append(test_reuses)
        unique_reused_list.append(test_uniques_reused)
        unique_list.append(test_unique)
    plot_reuse(comparison_name, test_names, reuse_list, unique_reused_list, unique_list)


def plot_reuse(comparison_name, test_names, reuse_list, unique_reused_list, unique_list):
    x = np.arange(len(test_names))
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(x + 0.00, reuse_list, label="#constraints reused", width=0.25)
    ax.bar(x + 0.25, unique_reused_list, label="#unique reused constraints", width=0.25)
    ax.bar(x + 0.5, unique_list, label="#unique constraints", width=0.25)
    ax.set_xticks(x)
    ax.set_xticklabels(test_names)
    plt.xticks(rotation=45)
    ax.legend()
    output_file_name = os.path.join(FIGURE_DIR, comparison_name, "reuses.png")
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
    plt.savefig(output_file_name, dpi=300, bbox_inches='tight')

def get_unique_set(constraints):
    return_set = set()
    for val in constraints:
        list = next(iter(val.values()))
        return_set.add((list[0], list[1]))
    return return_set

if __name__ == '__main__':
    calculate_and_plot_query_reuse("new_bar_graph", ["standard_COBRAS", "6", "9", "10"])
