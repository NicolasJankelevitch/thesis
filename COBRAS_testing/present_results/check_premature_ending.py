from config import FOLD_RESULT_DIR
import os
import ast

def check_premature_ending(test_names):
    for test_name in test_names:
        test_path = os.path.join(os.path.join(FOLD_RESULT_DIR, test_name), "clusterings")
        for dataset_name in os.listdir(test_path):
            dataset_folder = os.path.join(test_path, dataset_name)
            for f in os.listdir(dataset_folder):
                file = os.path.join(dataset_folder, f)
                with open(file) as d:
                    data = d.read()
                    dictionary = ast.literal_eval(data)
                    if dictionary["max_split_reached"] != 0:
                        count = len(dictionary["mls"]) + len(dictionary["cls"]) + len(dictionary["dks"])
                        print("{}\t{}\t{}\t{}".format(test_name, dataset_name, f, count))

if __name__ == '__main__':
    check_premature_ending(["standard_COBRAS", "6", "9", "10"])
