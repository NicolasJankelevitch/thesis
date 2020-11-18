import copy
import itertools
import os
import traceback

from sklearn.model_selection import KFold
import json
from config import FOLD_RESULT_DIR, COP_RF_LIKE_DIR
from generate_clusterings.Queriers.query_builders import ProbabilisticNoisyQuerierBuilder
from generate_clusterings.Queriers.queriers import FixedNoisyQuerier
from datasets import Dataset
from generate_clusterings.algorithms.my_mpckmeans import MyMPCKMeans
from generate_clusterings.algorithms.my_npu import NPU
from run_locally.run_tests import run_tests_from_generator

FOLD_SEEDS = [807610, 495307, 56867, 370993, 423374, 711836, 415552, 809578, 839429, 263647]
class TimesFoldAlgorithmRun:
    def __init__(self, algorithm, querier, dataset, train_indices, repeat_index, fold_index, test_name):
        self.algorithm = algorithm
        self.querier = querier
        self.dataset = dataset
        self.train_indices = train_indices
        self.repeat_index = repeat_index
        self.fold_index = fold_index
        self.test_name = test_name

    def get_full_result_path_filename(self):
        return os.path.join(FOLD_RESULT_DIR, self.test_name, "clusterings", self.dataset.name, "runNb_{}_fold_{}.txt".format(self.repeat_index, self.fold_index))


    def run(self):
        result = False
        while not result:
            try:
                all_clusters, runtimes, mls, cls = self.algorithm.fit(self.dataset.data, self.dataset.number_of_classes(), self.train_indices, self.querier)
                result = True
            except:
                print("ignored exception:")
                print(traceback.format_exc())
        path = self.get_full_result_path_filename()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, mode="w") as result_file:
            try:
                json.dump((all_clusters, runtimes, mls, cls, self.train_indices), result_file)
            except TypeError:
                print("all_clusters: ", type(all_clusters), type(all_clusters[0]), type(all_clusters[0][0]))
                print("runtimes: ", type(runtimes), type(runtimes[0]))
                print("ml", type(mls), type(mls[0]), type(mls[0][0]))
                print("cl", type(cls), type(cls[0]) )
                print("train_indices", type(self.train_indices), type(self.train_indices[0]))
                raise TypeError

class SingleTestAlgorithmRun:
    def __init__(self, algorithm, querier, dataset, query_percentage, noise_percentage, test_name):
        self.algorithm = algorithm
        self.querier = querier
        self.dataset = dataset
        self.query_percentage = query_percentage
        self.noise_percentage = noise_percentage
        self.test_name = test_name

    def get_full_result_path_filename(self):
        return os.path.join(COP_RF_LIKE_DIR, self.test_name, "q={}%,n={}%".format(self.query_percentage, self.noise_percentage), "{}.txt".format(self.dataset.name))

    def run(self):
        all_clusters, runtimes, mls, cls = self.algorithm.fit(self.dataset.data, self.dataset.number_of_classes(), range(self.dataset.number_of_instances()), self.querier)
        path = self.get_full_result_path_filename()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, mode="w") as result_file:
            json.dump((all_clusters[-1], runtimes[-1], mls, cls), result_file)


def generate_train_indices_for_nfold(n, seed, dataset):
    return [train.tolist() for train, test in KFold(n_splits = n, shuffle = True, random_state=seed).split(dataset.data)]

def test_active_clustering_algorithm_coprf_like(test_name, algorithm, constraint_ratios, noise_ratios, datasets, nb_cores = 3):
    print("calculating cop-rf like results for {}% constraints {}% noise for {}".format(constraint_ratios, noise_ratios,test_name))
    test_generator = generate_tests_active_clustering_algorithm_coprf_like(test_name, algorithm, constraint_ratios, noise_ratios, datasets)
    run_tests_from_generator(test_generator, nb_cores)


def test_active_clustering_algorithm_n_times_n_fold(test_name, algorithm, querier_builder, datasets, n =10, n_folds = 10, nb_cores = 3):
    print("calculating {}-times {}-fold clusterings for {}".format(n,n_folds, test_name))
    test_generator = generate_tests_active_clustering_algorithm_n_times_n_fold(test_name, algorithm, querier_builder, datasets, n, n_folds)
    run_tests_from_generator(list(test_generator), nb_cores)





def generate_tests_active_clustering_algorithm_coprf_like(test_name, algorithm, constraint_ratios, noise_ratios, datasets):
    for dataset in datasets:
        nb_instances = dataset.number_of_instances()
        max_constraints = nb_instances*(nb_instances-1)//2
        for constraint_ratio,noise_ratio in itertools.product(constraint_ratios, noise_ratios):
            query_limit = int(constraint_ratio/100*max_constraints)
            noisy_constraints = int(query_limit*noise_ratio/100)
            querier = FixedNoisyQuerier(dataset.target, noisy_constraints, query_limit)
            algorithm_copy = copy.deepcopy(algorithm)
            algorithm_run = SingleTestAlgorithmRun(algorithm_copy,querier, dataset, constraint_ratio, noise_ratio, test_name)
            if not os.path.isfile(algorithm_run.get_full_result_path_filename()):
                yield algorithm_run

def generate_tests_active_clustering_algorithm_n_times_n_fold(test_name, algorithm, querier_builder, datasets, n = 10, n_folds = 10):
    for dataset in datasets:
        yield from make_n_times_n_fold_scenario(test_name, algorithm, querier_builder, dataset, n, n_folds)


def make_n_times_n_fold_scenario(test_name, algorithm, querier_builder, dataset, n = 10, n_folds = 10):
    for repeat_index in range(n):
        fold_seed = FOLD_SEEDS[repeat_index]
        for fold_index, train_indices in enumerate(generate_train_indices_for_nfold(n_folds, fold_seed, dataset)):
            querier = querier_builder.build_querier(dataset)
            algorithm_copy = copy.deepcopy(algorithm)
            algorithm_run = TimesFoldAlgorithmRun(algorithm_copy, querier, dataset, train_indices, repeat_index, fold_index, test_name)
            if not os.path.isfile(algorithm_run.get_full_result_path_filename()):
                yield algorithm_run


if __name__ == '__main__':
    datasets = [Dataset("iris"), Dataset("ecoli")]
    clusterer = MyMPCKMeans()
    active_learner = NPU(clusterer=clusterer)
    test_active_clustering_algorithm_n_times_n_fold("results\\just_a_test", active_learner, ProbabilisticNoisyQuerierBuilder(0, 10), datasets, nb_cores=1)


