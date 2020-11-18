import os
import json
from util.config import FOLDS_RESULT_PATH
from util.datasets import Dataset
from test_cobras.evaluate_clustering.evaluation.ari import intermediate_results_to_ARIs, get_ARI
from test_cobras.run_tests import run_tests_from_generator


class CalculateLastARITask:
    def __init__(self, clustering_path, result_path, dataset_name):
        self.clustering_path = clustering_path
        self.result_path = result_path
        self.dataset_name = dataset_name

    def run(self):
        dataset = Dataset(self.dataset_name)
        target = dataset.target
        with open(self.clustering_path, mode = 'r') as clustering_file:
            clusterings, runtime, ml, cl, dn, train_indices = json.load(clustering_file)
        last_clustering = clusterings[-1]
        # Again the string "None" because JSON does not serialize None
        if train_indices == "None":
            ari = get_ARI(last_clustering, target)
        else:
            ari = get_ARI(last_clustering, target, train_indices=train_indices)

        os.makedirs(os.path.dirname(self.result_path), exist_ok=True)
        with open(self.result_path, mode='w') as ari_file:
            json.dump(ari, ari_file)


class CalculateIntermediateARIsTask:
    def __init__(self, clustering_path, result_path, dataset_name):
        self.clustering_path = clustering_path
        self.result_path = result_path
        self.dataset_name = dataset_name

    def run(self):
        dataset = Dataset(self.dataset_name)
        target = dataset.target
        with open(self.clustering_path, mode='r') as clustering_file:
            clusterings, runtimes, ml, cl, dn, train_indices = json.load(clustering_file)
        aris = intermediate_results_to_ARIs(clusterings, target, train_indices)
        os.makedirs(os.path.dirname(self.result_path), exist_ok=True)
        with open(self.result_path, mode='w') as ari_file:
            json.dump(aris, ari_file)


def generate_n_times_n_fold_ARI_tasks_for_tests(test_names, recalculate = False):
    for test_name in test_names:
        yield from generate_n_times_n_fold_ARI_tasks(test_name, recalculate)


def generate_n_times_n_fold_ARI_tasks(test_name, recalculate = False):
    for dataset_name in os.listdir(os.path.join(FOLDS_RESULT_PATH, test_name, "clusterings")):
        if dataset_name == ".DS_Store":
            continue
        dataset_path = os.path.join(FOLDS_RESULT_PATH, test_name, "clusterings", dataset_name)
        for result_file in os.listdir(dataset_path):
            clustering_path = os.path.join(FOLDS_RESULT_PATH, test_name, "clusterings", dataset_name, result_file)
            result_path = os.path.join(FOLDS_RESULT_PATH, test_name, "aris", dataset_name, result_file)
            if recalculate or not os.path.isfile(result_path):
                yield CalculateIntermediateARIsTask(clustering_path, result_path, dataset_name)


def calculate_n_times_n_fold_aris_for_testnames(test_names, nb_cores = 3, recalculate = False):
    task_generator = list(generate_n_times_n_fold_ARI_tasks_for_tests(test_names, recalculate))
    print("Calculating ARIs for n-times n-fold: ", test_names)
    run_tests_from_generator(task_generator, nb_cores)
