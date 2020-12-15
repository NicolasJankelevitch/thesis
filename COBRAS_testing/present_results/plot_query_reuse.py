from config import FOLD_RESULT_DIR
import os
import matplotlib.pyplot as plt
from config import FIGURE_DIR
import numpy as np


def calculate_and_plot_query_reuse(comparison_name, test_names):
    reuse_list = []
    attempt_list = []
    for test_name in test_names:
        test_path = os.path.join(FOLD_RESULT_DIR, test_name)
        test_path = os.path.join(test_path, "clusterings")
        total_test_reuses = 0
        total_test_attempts = 0
        for dataset_name in os.listdir(test_path):
            dataset_folder = os.path.join(test_path, dataset_name)
            total_dataset_reuses = 0
            total_dataset_attempts = 0
            for f in os.listdir(dataset_folder):
                file = os.path.join(dataset_folder, f)
                with open(file) as d:
                    data = d.read()
                    reuses = data[data.find("nr_reuses") + 11:]
                    reuses = reuses[:reuses.find(",")]
                    attempts = data[data.find("nr_tries")+10:-1]
                    total_dataset_reuses = total_dataset_reuses + int(reuses)
                    total_dataset_attempts = total_dataset_attempts + int(attempts)
            total_test_reuses = total_test_reuses + total_dataset_reuses
            total_test_attempts = total_test_attempts + total_dataset_attempts
        reuse_list.append(total_test_reuses)
        attempt_list.append(total_dataset_attempts)
    for r, a in zip(reuse_list, attempt_list):
        print(r/a)
    plot_reuse(comparison_name, test_names, reuse_list, attempt_list)


def plot_reuse(comparison_name, test_names, query_reuses, attempts):
    x = np.arange(len(test_names))
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(x + 0.00, query_reuses, label="query reuses", width=0.25)
    ax.bar(x + 0.25, attempts, label="all queries", width=0.25)
    ax.set_xticks(x)
    ax.set_xticklabels(test_names)
    plt.xticks(rotation=5)
    ax.legend()
    output_file_name = os.path.join(FIGURE_DIR, comparison_name, "reuses.png")
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
    plt.savefig(output_file_name, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    calculate_and_plot_query_reuse("new_bar_graph", ["standard_COBRAS", "6", "9", "10"])
