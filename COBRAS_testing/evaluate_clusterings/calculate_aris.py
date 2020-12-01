import sys
from json import JSONDecodeError
import jsonpickle
from pathlib import Path

from config import FOLD_RESULT_DIR, COP_RF_LIKE_DIR, RESULTS_PATH
import os
import json
from datasets import Dataset
from evaluate_clusterings.evaluation.ari import intermediate_results_to_ARIs,get_ARI
from run_locally.run_tests import run_tests_from_generator


def run_test(test):
    test.run()

class CalculateCOPRF_ARIsTask:
    def __init__(self, test_name, dataset_name):
        self.test_name = test_name
        self.dataset_name = dataset_name

    def run(self):
        dataset = Dataset(self.dataset_name)
        target = dataset.target
        directory = os.path.join(COP_RF_LIKE_DIR, self.test_name, 'clusterings', self.dataset_name)
        result_aris = dict()
        for run_name in os.listdir(directory):

            constraints, noise = run_name[:-4].split(",")
            # con_per = float(constraints[11:])
            noise_per = float(noise[5:])
            # "constraints{},noise{}.txt"
            full_path = os.path.join(directory, run_name)
            with open(full_path, mode='r') as clustering_file:
                clusterings, runtime, ml, cl, train_indices = json.load(clustering_file)
            last_clustering = clusterings[-1]
            if train_indices == "None":
                ari = get_ARI(last_clustering, target)
            else:
                ari = get_ARI(last_clustering, target, train_indices=train_indices)
            result_aris[noise_per] = ari
        ari_list_in_order = [v for k,v in sorted(result_aris.items())]
        result_file = os.path.join(COP_RF_LIKE_DIR,self.test_name, "aris", self.dataset_name+"_aris.txt")
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        with open(result_file, mode="w") as result:
            json.dump(ari_list_in_order, result)

class CalculateLastARITask:
    def __init__(self, clustering_path, result_path, dataset_name):
        self.clustering_path = clustering_path
        self.result_path = result_path
        self.dataset_name = dataset_name

    def run(self):
        dataset = Dataset(self.dataset_name)
        target = dataset.target
        with open(self.clustering_path, mode = 'r') as clustering_file:
            clusterings, runtime, ml, cl, train_indices = json.load(clustering_file)
        last_clustering = clusterings[-1]
        # Again the string "None" because JSON does not serialize None
        if train_indices == "None":
            ari = get_ARI(last_clustering, target)
        else:
            ari = get_ARI(last_clustering, target, train_indices=train_indices)

        os.makedirs(os.path.dirname(self.result_path), exist_ok=True)
        with open(self.result_path, mode='w') as ari_file:
            json.dump(ari, ari_file)

class CalculateIntermediateARIsTaskOLD:
    def __init__(self, clustering_path, result_path, dataset_name):
        self.clustering_path = clustering_path
        self.result_path = result_path
        self.dataset_name = dataset_name



    def run(self):
        dataset = Dataset(self.dataset_name)
        target = dataset.target
        with open(self.clustering_path, mode =  'r') as clustering_file:
            try:
                clusterings, runtimes, ml, cl, train_indices =json.load(clustering_file)
            except JSONDecodeError as e:
                print("error encountered in file {}".format(clustering_file), file = sys.stderr)
                raise e

        aris = intermediate_results_to_ARIs(clusterings, target, train_indices)
        os.makedirs(os.path.dirname(self.result_path), exist_ok=True)
        with open(self.result_path, mode = 'w') as ari_file:
            json.dump(aris, ari_file)

class CalculateIntermediateARIsTask:
    def __init__(self, clustering_path, result_path, dataset_name):
        self.clustering_path = clustering_path
        self.result_path = result_path
        self.dataset_name = dataset_name



    def run(self):
        dataset = Dataset(self.dataset_name)
        target = dataset.target
        file = Path(self.clustering_path)
        with file.open(mode='r') as input_file:
            string = input_file.readline()
            result_dir = jsonpickle.decode(string)
        clusterings = result_dir["clusterings"]
        train_indices = result_dir["train_indices"]

        aris = intermediate_results_to_ARIs(clusterings, target, train_indices)
        os.makedirs(os.path.dirname(self.result_path), exist_ok=True)
        with open(self.result_path, mode = 'w') as ari_file:
            json.dump(aris, ari_file)


def generate_coprf_like_ARI_tasks_for_tests(test_names):
    for test_name in test_names:
        yield from generate_coprf_like_ARI_tasks_for_test(test_name)

def generate_coprf_like_ARI_tasks_for_test(test_name):
    for dataset_name in os.listdir(os.path.join(COP_RF_LIKE_DIR, test_name, "clusterings")):
        for configuration in os.listdir(os.path.join(COP_RF_LIKE_DIR, test_name, "clusterings", dataset_name)):
            for task in os.listdir(os.path.join(COP_RF_LIKE_DIR, test_name, "clusterings", dataset_name, configuration)):
                clustering_path = os.path.join(COP_RF_LIKE_DIR, test_name, "clusterings", dataset_name, configuration, task)
                result_path = os.path.join(COP_RF_LIKE_DIR, test_name, "aris", dataset_name, configuration, task)
                if not os.path.isfile(result_path):
                    yield CalculateLastARITask(clustering_path, result_path, dataset_name)

def generate_n_times_n_fold_ARI_tasks_for_tests_OLD(test_names, recalculate = False):
    for test_name in test_names:
        yield from generate_n_times_n_fold_ARI_tasks_OLD(test_name, recalculate)

def generate_n_times_n_fold_ARI_tasks_OLD(test_name, recalculate = False):
    for dataset_name in os.listdir(os.path.join(FOLD_RESULT_DIR,test_name,"clusterings")):
        dataset_path = os.path.join(FOLD_RESULT_DIR,test_name, "clusterings", dataset_name)
        for result_file in os.listdir(dataset_path):
            clustering_path = os.path.join(FOLD_RESULT_DIR, test_name, "clusterings", dataset_name, result_file)
            result_path = os.path.join(FOLD_RESULT_DIR, test_name, "aris", dataset_name, result_file)
            if recalculate or not os.path.isfile(result_path):
                yield CalculateIntermediateARIsTaskOLD(clustering_path, result_path, dataset_name)

def calculate_coprf_like_aris_for_testnames(test_names, nb_cores = 3):
    task_generator = list(generate_coprf_like_ARI_tasks_for_tests(test_names))
    print("Calculating ARIs for cop-rf like: ", test_names)
    run_tests_from_generator(task_generator, nb_cores)

def calculate_n_times_n_fold_aris_for_testnames_OLD(test_names, nb_cores = 3, recalculate = False):
    task_generator = list(generate_n_times_n_fold_ARI_tasks_for_tests_OLD(test_names, recalculate))
    print("Calculating ARIs for n-times n-fold: ", test_names)
    run_tests_from_generator(task_generator, nb_cores)

def calculate_n_times_n_fold_aris_for_testnames(test_names, nb_cores = 3, recalculate = False):
    task_generator = list(generate_n_times_n_fold_ARI_tasks_for_tests(test_names, recalculate))
    print("Calculating ARIs for n-times n-fold: ", test_names)
    run_tests_from_generator(task_generator, nb_cores)

def generate_n_times_n_fold_ARI_tasks_for_tests(test_names, recalculate = False):
    for test_name in test_names:
        test_path = Path(FOLD_RESULT_DIR) / test_name
        result_path =  test_path / "clusterings"
        ari_path = test_path / "aris"
        for dataset_path in result_path.iterdir():
            dataset_name = dataset_path.name
            for clustering_path in dataset_path.iterdir():
                specific_ari_path = ari_path / dataset_name / clustering_path.name
                if recalculate or not specific_ari_path.exists():
                    yield CalculateIntermediateARIsTask(clustering_path, specific_ari_path, dataset_name)
