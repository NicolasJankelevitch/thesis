from config import FOLD_RESULT_DIR
import os
import ast


def check_split_levels_2(test_names):
    dataset_names = os.listdir(os.path.join(os.path.join(FOLD_RESULT_DIR, test_names[0]), "clusterings"))
    for dataset_name in dataset_names:
        for test_name in test_names:
            count = 0
            path = os.path.join(os.path.join(os.path.join(FOLD_RESULT_DIR, test_name), "clusterings"), dataset_name)
            for f in os.listdir(path):
                file = os.path.join(path, f)
                with open(file) as d:
                    data = d.read()
                    dictionary = ast.literal_eval(data)
                    count = count + sum(dictionary["split_levels"])
            print("{}\t{}\t{}".format(count, test_name, dataset_name))


def check_split_levels(test_names):
    for test_name in test_names:
        test_path = os.path.join(os.path.join(FOLD_RESULT_DIR, test_name), "clusterings")
        for dataset_name in os.listdir(test_path):
            dataset_folder = os.path.join(test_path, dataset_name)
            count = 0
            for f in os.listdir(dataset_folder):
                file = os.path.join(dataset_folder, f)
                with open(file) as d:
                    data = d.read()
                    dictionary = ast.literal_eval(data)
                    count = count + len(dictionary["split_levels"])
            print("{}\t{}\t{}".format(count, test_name, dataset_name))


if __name__ == '__main__':
    check_split_levels_2(["standard_COBRAS", "6", "9", "10"])
