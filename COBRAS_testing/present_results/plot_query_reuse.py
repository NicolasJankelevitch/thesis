from config import FOLD_RESULT_DIR
import os
import matplotlib.pyplot as plt
from config import FIGURE_DIR


def calculate_and_plot_query_reuse(comparison_name, test_names):
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    counter = 0
    for test_name in test_names:
        total_of_test = 0
        test_path = os.path.join(FOLD_RESULT_DIR, test_name)
        test_path = os.path.join(test_path, "clusterings")
        for dataset_name in os.listdir(test_path):
            total_of_dataset = 0
            dataset_folder = os.path.join(test_path, dataset_name)
            for f in os.listdir(dataset_folder):
                file = os.path.join(dataset_folder, f)
                with open(file) as d:
                    data = d.read()
                    reuses = data[data.find("nr_reuses") + 11:]
                    reuses = reuses[:reuses.find(",")]
                    total_of_dataset = total_of_dataset + int(reuses)
            total_of_test = total_of_test + total_of_dataset
        ax.bar(str(counter), total_of_test, label=test_name)
        ax.text(counter - 0.33, total_of_test / 2, str(total_of_test))
        counter = counter + 1
    ax.legend()
    output_file_name = os.path.join(FIGURE_DIR, comparison_name, "reuses.png")
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
    plt.savefig(output_file_name, dpi=300, bbox_inches='tight')
