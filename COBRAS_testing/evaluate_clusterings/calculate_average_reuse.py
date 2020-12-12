from config import FOLD_RESULT_DIR
import os
import matplotlib.pyplot as plt
from config import FIGURE_DIR


def calculate_and_plot_average_reuse(test_names, comparison_name):
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
            # print("{} has {} reused constraints".format(dataset_name, str(total_of_dataset)))
            total_of_test = total_of_test + total_of_dataset
        ax.bar(str(counter), total_of_test, label=test_name)
        ax.text(counter - 0.33, total_of_test / 2, str(total_of_test))
        counter = counter + 1
    ax.legend()
    output_file_name = os.path.join(FIGURE_DIR, comparison_name, "reuses.png")
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
    plt.savefig(output_file_name, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    # string = "nr_reuses: 707, nr_tries: 32313"
    # reuses = string[string.find("nr_reuses")+11:]
    # reuses = reuses[:reuses.find(",")]
    # tries = string[string.find("nr_tries")+10:]
    # print(reuses)
    # print(tries)
    calculate_and_plot_average_reuse(["strat_1", "strat_1_with_standard_kmeans", "strat_2",
                                      "strat_2_with_standard_kmeans", "basic_COBRAS", "strat_2_with_selection",
                                      "strat_2_with_selection_and_standard_kmeans"], "test_reuse_graph")
