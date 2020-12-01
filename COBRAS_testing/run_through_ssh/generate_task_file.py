import os
import random

import numpy as np

from config import FOLD_PATH, FOLD_RESULT_DIR, TASK_PATH, COP_RF_LIKE_DIR
from datasets import Dataset

"""
The task file should look as follows (no spaces allowed in paths):
<algorithm name> <dataset-name> (<path to fold> OR "no_test_set") <path to clustering file for the result> \n
<querier name> <querier parameters> \n
<algorithm specific parameters> \n
"""


def get_fold_path(dataset_name, runnb, foldnb):
    assert 0 <= foldnb <= 9
    assert 0 <= runnb <= 9
    return os.path.join(FOLD_PATH, dataset_name, "run{}_fold{}.txt".format(runnb, foldnb))

def get_task_path(machine_name):
    return os.path.join(TASK_PATH, machine_name)

def mpck_means_algorithm_settings_to_string(w = 1, max_iter = 10, learn_full = False):
    return " ".join(str(x) for x in [w, max_iter, learn_full])

def cobras_algorithm_settings_to_string(noise_probability=0.10, min_approx_order=2, max_approx_order=4,
                                        keep_threshold=0.91, reuse_threshold=0.91, correct_noise=True, use_all_cycles  = False):
    return " ".join(str(x) for x in [noise_probability,min_approx_order,max_approx_order,keep_threshold,reuse_threshold,correct_noise,use_all_cycles])

def fixed_amount_of_noise_querier_settings_to_string(number_of_noisy_constraints = 10, max_queries=100):
    return "{} {}".format(number_of_noisy_constraints, max_queries)

def probabilistic_noisy_querier_settings_to_string(noise_probability=0.10, max_queries=100):
    return "{} {}".format(noise_probability, max_queries)


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

    def add_cop_rf_like_tests_for_cobras_varying_noise(self, test_name, datasets, constraint_percentage, noise_percentages, nb_runs = 10, min_approx_order = 2, max_approx_order = 4):
        querier_name = "fixed_noise_querier"
        algorithm_name = "COBRAS"
        for dataset_name in datasets:
            for noise_percentage in noise_percentages:

                dataset  = Dataset(dataset_name)
                nb_instances = dataset.number_of_instances()
                nb_constraints = int((nb_instances*(nb_instances-1)/2)*constraint_percentage)
                nb_noise = int(nb_constraints*noise_percentage)
                querier_parameters = fixed_amount_of_noise_querier_settings_to_string(nb_noise, nb_constraints)
                threshold = 0.98
                algorithm_parameters = cobras_algorithm_settings_to_string(noise_percentage,min_approx_order,max_approx_order, keep_threshold=threshold, reuse_threshold=threshold,correct_noise=True)
                fold_path = "no_test_set"
                for run_nb in range(nb_runs):
                    result_file_path = os.path.join(COP_RF_LIKE_DIR, test_name, "clusterings", dataset_name, "c{},n{}".format(constraint_percentage,noise_percentage),"run{}.txt".format(run_nb))
                    if not os.path.isfile(result_file_path):
                        self.add_test(algorithm_name, dataset_name, result_file_path, querier_name, querier_parameters, algorithm_parameters, fold_path)



    def add_10_times_10_fold_test(self, test_name, algorithm_name, algorithm_parameters, datasets, querier_name,
                                  querier_parameters, nb_of_runs = 10):
        settings_file_path = os.path.join(FOLD_RESULT_DIR, test_name, "settings.txt")
        os.makedirs(os.path.dirname(settings_file_path), exist_ok=True)
        # to make sure we know what parameters were used for a certain test_cobras run
        with open(settings_file_path, mode="w") as settings_file:
            settings_file.write("{} {}\n".format(algorithm_name, algorithm_parameters))
            settings_file.write("{} {}\n".format(querier_name, querier_parameters))
        for dataset_name in datasets:
            for run_nb in range(nb_of_runs):
                for fold_nb in range(10):
                    fold_path = get_fold_path(dataset_name, run_nb, fold_nb)
                    resulting_file_name = os.path.join(FOLD_RESULT_DIR, test_name, "clusterings", dataset_name,
                                                       "run{}_fold{}.txt".format(run_nb, fold_nb))
                    if not os.path.isfile(resulting_file_name):
                        self.add_test(algorithm_name, dataset_name, resulting_file_name, querier_name, querier_parameters,
                                  algorithm_parameters, fold_path)

    def divide_over_task_files(self, machine_names):
        #equally distributes tests over machines
        random.shuffle(self.test_list)
        chunks = np.array_split(self.test_list, len(machine_names))
        for chunk, machine_name in zip(chunks, machine_names):
            file_name = get_task_path(machine_name)
            os.makedirs(os.path.dirname(file_name), exist_ok= True)
            with open(file_name, mode = "w") as task_file:
                for test in chunk:
                    task_file.write(test)

