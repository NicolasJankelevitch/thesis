import json
import os
import sys
import traceback

from config import TASK_PATH, COBRAS_PATH
from generate_clusterings.algorithms.my_cosc import MyCOSCMatlab
from generate_clusterings.algorithms.my_mpckmeans_java import MyMPCKMeansJava
from generate_clusterings.algorithms.my_npu import NPU

sys.path.append(COBRAS_PATH)
from cobras_ts.cobras import COBRAS

from datasets import Dataset
from generate_clusterings.Queriers.query_builders import ProbabilisticNoisyQuerierBuilder, FixedNoisyQuerierBuilder
from run_locally.run_tests import run_tests_from_generator


def fold_path_to_train_indices(fold_path):
    if fold_path == "no_test_set":
        return None
    else:
        with open(fold_path, mode='r') as file:
            indices_dict = json.load(file)
        train_indices = indices_dict["train_indices"]
        return train_indices

def semi_sup_algo_name_to_object(algorithm_name, algorithm_parameters):
    if algorithm_name == "MPCKmeans":
        w, max_iter, learn_full = algorithm_parameters
        w = float(w)
        max_iter = int(max_iter)
        learn_full = learn_full == "True"
        return MyMPCKMeansJava(max_iter)
    elif algorithm_name == "COSC":
        return MyCOSCMatlab()
    else:
        raise Exception("unknown semi_supp algo name")

def algorithm_info_to_object(algorithm_name, algorithm_parameters):
    if algorithm_name == "COBRAS":
        noise_probability, min_approx_order, max_approx_order, keep_threshold, reuse_threshold, correct_noise, use_all_cycles = algorithm_parameters
        noise_probability = float(noise_probability)
        min_approx_order = int(min_approx_order)
        max_approx_order = int(max_approx_order)
        keep_threshold = float(keep_threshold)
        reuse_threshold = float(reuse_threshold)
        correct_noise = correct_noise == "True"  # because pythons bool("False") = True
        use_all_cycles = use_all_cycles == "True"
        return COBRAS(noise_probability=noise_probability, minimum_approximation_order=min_approx_order,
                      maximum_approximation_order=max_approx_order, certainty_threshold_keep=keep_threshold,
                      certainty_threshold_reuse=reuse_threshold, correct_noise=correct_noise)
    elif "NPU" in algorithm_name:
        semi_sup_algo_name = algorithm_name[4:]
        semi_sup_algo = semi_sup_algo_name_to_object(semi_sup_algo_name, algorithm_parameters)
        clusterer = NPU(semi_sup_algo)
        return clusterer
    else:
        raise Exception("unknown algorithm name!")


def querier_info_to_object(querier_name, querier_parameters):
    if querier_name == "probability_noise_querier":
        noise_probability, max_queries = querier_parameters
        noise_probability = float(noise_probability)
        max_queries = int(max_queries)
        return ProbabilisticNoisyQuerierBuilder(noise_probability, max_queries)
    elif querier_name == "fixed_noise_querier":
        nb_noisy, max_queries = querier_parameters
        nb_noisy = int(nb_noisy)
        max_queries = int(max_queries)
        return FixedNoisyQuerierBuilder(max_queries,nb_noisy)
    else:
        raise Exception("unknown querier name")


def general_run_task_from_lines(lines):
    algorithm_name, dataset_name, fold_path, result_path = lines[0].split(" ")
    querier_info = lines[1].split(" ")
    querier_name = querier_info[0]
    if len(querier_info) > 1:
        querier_parameters = querier_info[1:]
    else:
        querier_parameters = []
    algorithm_parameters = lines[2].split(" ")
    return GeneralRunTask(algorithm_name, dataset_name, fold_path, result_path, querier_name,querier_parameters, algorithm_parameters)


class GeneralRunTask:
    def __init__(self, algorithm_name, data_name, fold_path, result_path, querier_name, querier_parameters, algorithm_parameters):
        self.algorithm_name = algorithm_name
        self.dataset_name = data_name
        self.fold_path = fold_path
        self.querier_name = querier_name
        self.querier_parameters = querier_parameters
        self.algorithm_parameters = algorithm_parameters
        self.result_path = result_path


    def run(self):
        algorithm = algorithm_info_to_object(self.algorithm_name, self.algorithm_parameters)
        querier_builder = querier_info_to_object(self.querier_name, self.querier_parameters)
        dataset = Dataset(self.dataset_name)
        train_indices = fold_path_to_train_indices(self.fold_path)
        querier = querier_builder.build_querier(dataset)
        result = None

        # retry to execute the algorithm 10 times
        # this is because COSC does not always produce a result and ends with an exception
        try:
            result = algorithm.fit(dataset.data, dataset.number_of_classes(), train_indices, querier)
        except Exception as e:
            print("An exception occured during calculation of {} (this is silently ignored):".format(self.result_path), file = sys.stderr)
            traceback.print_exc()

        if result is None:
            return

        # None is not json serializable so use the string "None" instead
        train_indices = train_indices if train_indices is not None else "None"
        full_result = result + (train_indices,)
        os.makedirs(os.path.dirname(self.result_path), exist_ok=True)
        with open(self.result_path, mode="w") as result_file:
            json.dump(full_result, result_file)


def run_task_file_locally(task_file_name, nb_cores):
    file_path = os.path.join(TASK_PATH, task_file_name)
    with open(file_path, mode="r") as task_file:
        tasks = []
        end_reached = False
        while not end_reached:
            lines = []
            for i in range(3):
                next_line = task_file.readline()
                if next_line is None or len(next_line) == 0:
                    if i == 0:
                        end_reached = True
                        break
                    else:
                        raise Exception("unexpected end of file!")
                lines.append(next_line.strip())
            if not end_reached:
                tasks.append(general_run_task_from_lines(lines))

    run_tests_from_generator(tasks, nb_of_cores=nb_cores)


if __name__ == '__main__':
    sys.setrecursionlimit(5000)
    _, task_file_name, nb_cores = sys.argv
    run_task_file_locally(task_file_name, int(nb_cores))
    # run_task_file("machine1.txt",1)
