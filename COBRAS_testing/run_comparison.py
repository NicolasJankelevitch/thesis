"""
    I think I can just delete this!
"""
import winsound

#from COBRAS_dont_know.cobras import COBRAS

from datasets import Dataset
from generate_clusterings.Queriers.before_clustering_queriers import RandomQuerierBuilder
from generate_clusterings.Queriers.query_builders import ProbabilisticNoisyQuerierBuilder, FixedNoisyQuerierBuilder
from generate_clusterings.active_testing import test_active_clustering_algorithm_n_times_n_fold, test_active_clustering_algorithm_coprf_like
from evaluate_clusterings.calculate_aris import calculate_n_times_n_fold_aris_for_testnames_OLD
from evaluate_clusterings.calculate_area_under_ARI import calculate_area_under_ari_for
from evaluate_clusterings.calculate_average_aris import calculate_average_aris
from evaluate_clusterings.calculate_aligned_rank import calculate_and_write_aligned_rank
from generate_clusterings.algorithms.dummy_clusterer import DummyClusterer
from generate_clusterings.algorithms.my_kmeans import KMeansClusterer
from generate_clusterings.algorithms.my_pckmeans import MyPCKMeans
from generate_clusterings.algorithms.my_query_constraints_first_clusterer import QueryFirstActiveClusterer
from present_results.plot_aris import plot_overall_average_ARI, plot_average_ARI_per_dataset
from present_results.plot_aligned_rank import plot_rank_comparison_file
from generate_clusterings.algorithms.my_mpckmeans import MyMPCKMeans
from generate_clusterings.algorithms.my_npu import NPU
from multiprocessing import freeze_support
from run_through_ssh.generate_task_file import *
from run_through_ssh.generate_task_file import TestCollection


def make_sound():
    for i in range(3):
        winsound.PlaySound('SystemAsterix', winsound.SND_ALIAS)

def compare_algorithms_and_plot_results_n_times_n_fold(comparison_name, algorithms, line_names = None, budget = 100, calculate_aris = True):
    if calculate_aris:
        calculate_n_times_n_fold_aris_for_testnames_OLD(algorithms)
        calculate_average_aris(algorithms, budget)
    if line_names is None:
        line_names = algorithms
    print("calculating rank comparison file")
    calculate_and_write_aligned_rank(algorithms, comparison_name)
    print("plotting ARI comparisons")
    plot_average_ARI_per_dataset(comparison_name, algorithms, line_names)
    plot_overall_average_ARI(comparison_name,algorithms, line_names)
    print("plotting aligned rank comparison")
    plot_rank_comparison_file(comparison_name, algorithms, line_names)


def run_coprflike_tests():
    dataset_names = [
        "iris",
        # "parkinsons",
        "glass",
        "ionosphere",
        # "segmentation",
    ]
    coprf_datasets = [Dataset(name, preprocessed=False) for name in dataset_names]
    constraint1, noise1 = [0.1, 0.2, 0.3, 0.4, 0.5], [15]
    constraints2, noise2 = [0.3], [5, 10, 15, 20, 25, 30]
    clusterer = COBRAS(noise_probability=0.20, minimum_approximation_order=3, maximum_approximation_order=6, certainty_threshold_keep=0.85, certainty_threshold_reuse=0.90)
    test_active_clustering_algorithm_coprf_like("nCOBRAS_p0.20,O(3,6),t(0.85,0.90)", clusterer, constraint1, noise1,
                                                coprf_datasets)
    calculate_all_coprf_aris(nb_cores=1)
    calculate_area_under_ari_for(["nCOBRAS_p0.20,O(3,6),t(0.85,0.90)"],nb_of_cores=1)


def PCK_means_noise_vs_no_noise():
    names = ["breast-cancer-wisconsin","column_2C", "dermatology", "ecoli", "glass", "hepatitis", "ionosphere", "iris"]
    datasets = [Dataset(name) for name in names]
    clusterer = QueryFirstActiveClusterer(MyMPCKMeans(learn_multiple_full_matrices=True))
    test_active_clustering_algorithm_n_times_n_fold("random_MPCK_means_full_no_noise", clusterer, RandomQuerierBuilder(None, 100, 0),datasets, nb_cores=4, n=1)
    test_active_clustering_algorithm_n_times_n_fold("random_MPCK_means_full_noise",
                                                    clusterer,
                                                    RandomQuerierBuilder(None, 100, 0.15), datasets, nb_cores=4, n=1)
    calculate_all_aris()
    calculate_all_average_aris()
    comparison_name = "full MPCK-means noise vs no noise"
    algorithms = ["random_MPCK_means_full_no_noise", "random_MPCK_means_full_noise"]
    compare_algorithms_and_plot_results_n_times_n_fold(comparison_name, algorithms)

def run_tests():

    datasets = [Dataset(name) for name in Dataset.get_non_face_news_spam_names()]
    test_active_clustering_algorithm_n_times_n_fold("nCOBRAS_filtering", COBRAS(minimum_approximation_order=3, noise_probability=0.10, maximum_approximation_order=7,certainty_threshold_reuse=0.91), ProbabilisticNoisyQuerierBuilder(0.10, 100), datasets, n=3)

    # test_active_clustering_algorithm_n_times_n_fold("NPU-MPCK-means_no_noise", NPU(MyMPCKMeans(max_iter=10, learn_multiple_full_matrices=False)), NoisyQuerierBuilder(0, 100), datasets, nb_cores=3)
    # test_active_clustering_algorithm_n_times_n_fold("NPU-MPCK-means_10%_noise", NPU(MyMPCKMeans(max_iter = 10, learn_multiple_full_matrices=False)), NoisyQuerierBuilder(0.10, 100), datasets, nb_cores=3)
    # test_active_clustering_algorithm("dummy_clusterer1", DummyClusterer(), NoisyQuerierBuilder(0.10, 100), datasets )
    # test_active_clustering_algorithm("dummy_clusterer2", DummyClusterer(), NoisyQuerierBuilder(0.10, 100), datasets)
    test_names = ["nCOBRAS_filtering", "noise_robust_cobras"]

    compare_algorithms_and_plot_results_n_times_n_fold("nCOBRAS_filtering", test_names, calculate_aris=False)




def dummy_test():
    datasets = [Dataset("iris")]
    clusterer = COBRAS(correct_noise=True, noise_probability=0.05,certainty_threshold_keep=0.96, certainty_threshold_reuse=0.96)
    test_active_clustering_algorithm_n_times_n_fold("noise_robust_COBRAS",clusterer,ProbabilisticNoisyQuerierBuilder(0.05,100), datasets)
    clusterer = COBRAS(correct_noise = False, noise_probability=0.05, certainty_threshold_keep=0.96, certainty_threshold_reuse=0.96)
    test_active_clustering_algorithm_n_times_n_fold("COBRAS_noise", clusterer, ProbabilisticNoisyQuerierBuilder(0.05,100), datasets)
    clusterer = COBRAS(correct_noise= False, noise_probability=0.05, certainty_threshold_keep=0.96, certainty_threshold_reuse=0.96)
    test_active_clustering_algorithm_n_times_n_fold("COBRAS_no_noise", clusterer, ProbabilisticNoisyQuerierBuilder(0,100), datasets)
    calculate_all_aris()
    calculate_all_average_aris()
    compare_algorithms_and_plot_results_n_times_n_fold("test_cobras", ["noise_robust_COBRAS","COBRAS_no_noise", "COBRAS_noise"])

if __name__ == '__main__':
    # calculate_all_aris(4)
    freeze_support()
    run_tests()
    # PCK_means_noise_vs_no_noise()
    # run_coprflike_tests()
    # dummy_test()
    # dummy_test()
    # make_sound()