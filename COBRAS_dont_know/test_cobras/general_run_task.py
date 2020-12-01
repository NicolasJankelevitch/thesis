import json
import os
import sys
from cobras.querier.weak_querier import WeakQuerier, WeakQuerierBuilder
from util.config import TASKS_PATH
from cobras.cobras import COBRAS
from util.datasets import Dataset
from test_cobras.run_tests import run_tests_from_generator


def fold_path_to_train_indices(fold_path):
    if fold_path == "no_test_set":
        return None
    else:
        with open(fold_path, mode='r') as file:
            indices_dict = json.load(file)
        train_indices = indices_dict["train_indices"]
        return train_indices


def algorithm_info_to_object(algorithm_name, algorithm_parameters=None):
    if algorithm_name == "COBRAS":
        return COBRAS()
    else:
        raise Exception("unknown algorithm name!")


def querier_info_to_object(querier_name, querier_parameters):
    if querier_name == "weak_querier":
        oracle_type, q, rho, nu, max_prob, max_queries = querier_parameters
        oracle_type = str(oracle_type)
        q = float(q)
        rho = float(rho)
        nu = float(nu)
        max_prob = float(max_prob)
        max_queries = int(max_queries)
        return WeakQuerierBuilder(oracle_type, q, rho, nu, max_prob, max_queries)
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

        result = algorithm.fit(dataset.data, dataset.number_of_classes(), train_indices, querier)

        save_pred_results = False
        if save_pred_results:
            logger = algorithm.logger
            stats = (querier.total_DK, len(logger.predicted_constraints), logger.n_correct_preds)
            output_file = '' + self.result_path[17:]
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, mode="w") as result_file:
                json.dump(stats, result_file)

        save_rf_accuracy = False
        if save_rf_accuracy:
            output_file = '' + self.result_path[17:]
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, mode="w") as result_file:
                json.dump(algorithm.logger.accuracy_per_n_constraints, result_file)

        # None is not json serializable so use the string "None" instead
        train_indices = train_indices if train_indices is not None else "None"
        full_result = result + (train_indices,)
        os.makedirs(os.path.dirname(self.result_path), exist_ok=True)
        with open(self.result_path, mode="w") as result_file:
            json.dump(full_result, result_file)


def run_task_file(task_file_name, nb_cores):
    file_path = os.path.join(TASKS_PATH, task_file_name)
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
