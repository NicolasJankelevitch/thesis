import os
import random
import numpy as np
from util.config import FOLDS_PATH, FOLDS_RESULT_PATH, TASKS_PATH
"""
The task file should look as follows (no spaces allowed in paths):
<algorithm name> <dataset-name> (<path to fold> OR "no_test_set") <path to clustering file for the result> \n
<querier name> <querier parameters> \n
<algorithm specific parameters> \n
"""


def get_fold_path(dataset_name, runnb, foldnb):
    assert 0 <= foldnb <= 9
    assert 0 <= runnb <= 9
    return os.path.join(FOLDS_PATH, dataset_name, "run{}_fold{}.txt".format(runnb, foldnb))


def get_task_path(machine_name):
    return os.path.join(TASKS_PATH, machine_name)


def cobras_algorithm_settings_to_string():
    return ""


def weak_querier_settings_to_string(oracle_type='random', q=1, rho=0.8, nu=1.25, max_prob=0.95, max_queries=100):
    return "{} {} {} {} {} {}".format(oracle_type, q, rho, nu, max_prob, max_queries)


class TestCollection:
    def __init__(self):
        # a list of strings each string representing a test_cobras in the format listed above
        self.test_list = []
        self.resulting_file_names = []

    def add_test(self, algorithm_name, dataset_name, resulting_file, querier_name, querier_parameters,
                 algorithm_parameters, fold="no_test_set"):
        self.resulting_file_names.append(resulting_file)
        test_string = \
            "{} {} {} {}\n".format(algorithm_name,dataset_name,fold, resulting_file) +\
                "{} {}\n".format(querier_name, querier_parameters) +\
                "{}\n".format(algorithm_parameters)
        self.test_list.append(test_string)

    def add_10_times_10_fold_test(self, test_name, algorithm_name, algorithm_parameters, datasets, querier_name,
                                  querier_parameters, nb_of_runs=10):
        settings_file_path = os.path.join(FOLDS_RESULT_PATH, test_name, "settings.txt")
        os.makedirs(os.path.dirname(settings_file_path), exist_ok=True)
        # to make sure we know what parameters were used for a certain test_cobras run
        with open(settings_file_path, mode="w") as settings_file:
            settings_file.write("{} {}\n".format(algorithm_name, algorithm_parameters))
            settings_file.write("{} {}\n".format(querier_name, querier_parameters))
        for dataset_name in datasets:
            for run_nb in range(nb_of_runs):
                for fold_nb in range(10):
                    fold_path = get_fold_path(dataset_name, run_nb, fold_nb)
                    resulting_file_name = os.path.join(FOLDS_RESULT_PATH, test_name, "clusterings", dataset_name,
                                                       "run{}_fold{}.txt".format(run_nb, fold_nb))
                    if not os.path.isfile(resulting_file_name):
                        self.add_test(algorithm_name, dataset_name, resulting_file_name, querier_name, querier_parameters,
                                  algorithm_parameters, fold_path)

    def divide_over_task_files(self, machine_names):
        # Equally distributes tests over machines
        random.shuffle(self.test_list)
        chunks = np.array_split(self.test_list, len(machine_names))
        for chunk, machine_name in zip(chunks, machine_names):
            file_name = get_task_path(machine_name)
            os.makedirs(os.path.dirname(file_name), exist_ok= True)
            with open(file_name, mode="w") as task_file:
                for test in chunk:
                    task_file.write(test)
