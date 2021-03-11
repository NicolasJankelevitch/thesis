import ast
import matplotlib.pyplot as plt
import numpy as np
import os

from config import FOLD_RESULT_DIR, FIGURE_DIR


def plot_overall_predictions(comparison_name, test_names):
    cl_list = []
    ml_list = []
    dk_list = []
    for test_name in test_names:
        test_path = os.path.join(os.path.join(FOLD_RESULT_DIR, test_name), "clusterings")
        cls = 0
        mls = 0
        dks = 0
        for dataset_name in os.listdir(test_path):
            dataset_folder = os.path.join(test_path, dataset_name)
            for f in os.listdir(dataset_folder):
                file = os.path.join(dataset_folder, f)
                #print(file)
                with open(file) as d:
                    data = d.read()
                    dictionary = ast.literal_eval(data)
                    for tup in dictionary["predicted"]:
                        constr = extract_constr(tup)
                        if constr == "CL":
                            cls += 1
                        elif constr == "ML":
                            mls += 1
                        elif constr == "DK":
                            dks += 1
        print("{}:\n\tCLS: {}\n\tMLS: {}\n\tDKS: {}\n\n".format(test_name, cls, mls, dks))
        #cl_list.append(cls)
        #ml_list.append(mls)
        #dk_list.append(dks)
    #output_file_name = os.path.join(FIGURE_DIR, comparison_name, "assignments")
    #draw_plot(output_file_name, test_names, cl_list, ml_list, dk_list)


def draw_plot(output_file_name, test_names, cls, mls, dks):
    x = np.arange(len(test_names))
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(x - 0.25, cls, label="cls", width=0.25)
    ax.bar(x + 0.00, mls, label="mls", width=0.25)
    ax.bar(x + 0.25, dks, label="dks", width=0.25)
    for i, v in enumerate(cls):
        plt.text(i - .375, v, "CLS: " + str(v))
    for i, v in enumerate(mls):
        plt.text(i - 0.125, v, "MLS: " + str(v))
    for i, v in enumerate(dks):
        plt.text(i + .125, v, "DLS: " + str(v))
    ax.set_xticks(x)
    ax.set_xticklabels(test_names)
    plt.xticks(rotation=45)
    ax.legend()
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
    plt.savefig(output_file_name, dpi=300, bboxinches='tight')
    plt.close(fig)


def extract_constr(tup):
    return tup["py/tuple"][2]


def plot_predictions_per_dataset(comparison_name, test_names):
    datasets = os.listdir(os.path.join(FOLD_RESULT_DIR, test_names[0], "clusterings"))
    for dataset in datasets:
        cl_list = []
        ml_list = []
        dk_list = []
        for test_name in test_names:
            cls = 0
            mls = 0
            dks = 0
            path = os.path.join(os.path.join(os.path.join(FOLD_RESULT_DIR, test_name), "clusterings"), dataset)
            for f in os.listdir(path):
                file = os.path.join(path, f)
                with open(file) as d:
                    data = d.read()
                    dictionary = ast.literal_eval(data)
                    for tup in dictionary["predicted"]:
                        constr = extract_constr(tup)
                        if constr == "CL":
                            cls += 1
                        elif constr == "ML":
                            mls += 1
                        elif constr == "DK":
                            dks += 1
            cl_list.append(cls)
            ml_list.append(mls)
            dk_list.append(dks)
        output_file_name = os.path.join(FIGURE_DIR, comparison_name, "assignments_per_test", dataset)
        draw_plot(output_file_name, test_names, cl_list, ml_list, dk_list)


if __name__ == '__main__':
    plot_overall_predictions("intra_pred_check_preds_update1", ["KEEP_intra_pred_max_10fold",
                                                                "KEEP_intra_pred_min_10fold",
                                                                "KEEP_inter_pred_max_10fold",
                                                                "KEEP_inter_pred_min_10fold"])
