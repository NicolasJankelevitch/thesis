from cobras_ts.cobras import COBRAS
from datasets import Dataset
from evaluate_clusterings.calculate_aligned_rank import calculate_and_write_aligned_rank
from evaluate_clusterings.calculate_aris import calculate_n_times_n_fold_aris_for_testnames
from evaluate_clusterings.calculate_average_aris import calculate_average_aris
from evaluate_clusterings.calculate_average_runtimes import calculate_average_runtimes_for_tests
from generate_clusterings.clustering_task import make_n_run_10_fold_cross_validation, cobras_result_extractor, cosc_result_extractor
from present_results.plot_aligned_rank import plot_rank_comparison_file
from present_results.plot_aris import plot_overall_average_ARI, plot_average_ARI_per_dataset
from present_results.plot_runtimes import plot_runtime_comparison
from querier.noisy_labelquerier import ProbabilisticNoisyQuerier
from run_with_dask.run_with_dask import execute_list_of_clustering_tasks

COLORS_TO_USE = [u'#2ca02c', u'#ff7f0e', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22',
                 u'#17becf']


def calculate_aris_and_compare_for_tests(comparison_name, test_names, line_names=None, nb_of_cores=8, recalculate=False,
                                         offsets_dict=None, query_budget=None):
    calculate_n_times_n_fold_aris_for_testnames(test_names, nb_cores=nb_of_cores, recalculate=recalculate)
    calculate_average_aris(test_names, query_budget)
    # calculate_average_runtimes_for_tests(test_names, query_budget, recalculate=recalculate)
    compare_algorithms_and_plot_results_n_times_n_fold(comparison_name, test_names, query_budget, line_names,
                                                       offset_dict=offsets_dict)


def compare_algorithms_and_plot_results_n_times_n_fold(comparison_name, algorithms, query_budget, line_names=None,
                                                       offset_dict=None, ):
    if line_names is None:
        line_names = algorithms
    print("calculating rank comparison file")
    calculate_and_write_aligned_rank(algorithms, comparison_name)
    print("plotting ARI comparisons")
    plot_average_ARI_per_dataset(comparison_name, algorithms, line_names)
    plot_overall_average_ARI(comparison_name, algorithms, line_names, offset_dict=offset_dict)
    standard_colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f',
                       u'#bcbd22', u'#17becf']
    colors_to_use = COLORS_TO_USE
    # plot_runtime_comparison(comparison_name,algorithms, line_names, colors_to_use, query_budget)
    print("plotting aligned rank comparison")
    plot_rank_comparison_file(comparison_name, algorithms, line_names)


def compare_clustering_quality_all_varying_amounts_of_noise():
    query_budget = 200
    test_names_with_noise_hole = [
        "new_cobras_{}_budget_200",
        "new_ncobras_{}_budget_200",
        "new_NPU_COSC_{}_budget_200",
    ]

    for noise_percentage in ["no", "0.05", "0.1"]:
        test_names = [test_name.format(noise_percentage) for test_name in test_names_with_noise_hole]
        calculate_aris_and_compare_for_tests(
            "new_varying_amounts_of_noise_{}_budget_{}".format(noise_percentage, query_budget), test_names,
            recalculate=False, nb_of_cores=4, query_budget=query_budget)


def compare_clustering_quality_all_varying_amounts_of_noise_OLD():
    query_budget = 200
    # test_names_with_noise_hole= [
    #     "new_cobras_{}_budget_200",
    #     "cobras_{}_noise_budget200",
    #     "new_ncobras_{}_budget_200",
    #     "NCOBRASplus_{}_noise_budget200_pnoise0.05_threshold0.95",
    #     "NPU_Cosc_{}_noise_budget200",
    #     "NPU_MPCKmeans_{}_noise_budget200",
    # ]
    line_names = ["nCOBRAS", "COBRAS", "NPU_Cosc", "NPU_MPCKmeans"]
    no_noise = ['new_ncobras_no_budget_200', 'new_cobras_no_budget_200',
                'NPU_Cosc_no_noise_budget200',
                'NPU_MPCKmeans_no_noise_budget200']

    low_noise = ['new_ncobras_0.05_budget_200', 'new_cobras_0.05_budget_200',
                 'NPU_Cosc_0.05_noise_budget200',
                 'NPU_MPCKmeans_0.05_noise_budget200']
    high_noise = ['new_ncobras_0.1_budget_200', 'new_cobras_0.1_budget_200',
                  'NPU_Cosc_0.1_noise_budget200',
                  'NPU_MPCKmeans_0.1_noise_budget200']
    for noise_percentage, test_names in zip(["no", "0.05", "0.1"], [no_noise, low_noise, high_noise]):
        calculate_aris_and_compare_for_tests(
            "new_varying_amounts_of_noise_{}_budget_{}".format(noise_percentage, query_budget), test_names, line_names,
            recalculate=False, nb_of_cores=4, query_budget=query_budget)


def compare_runtimes():
    for query_budget in [100, 200]:
        low_noise = ['new_ncobras_0.05_budget_200',
                     'new_cobras_0.05_budget_200',
                     'NPU_Cosc_0.05_noise_budget200',
                     'NPU_MPCKmeans_0.05_noise_budget200']
        line_names = ["nCOBRAS", "COBRAS", "NPU_Cosc", "NPU_MPCKmeans"]
        calculate_average_runtimes_for_tests(low_noise[2:], query_budget, use_old=True)
        calculate_average_runtimes_for_tests(low_noise[:2], query_budget, use_old=False)
        plot_runtime_comparison("new_runtime_comparison", low_noise, line_names, COLORS_TO_USE, query_budget)


def compare_cobras_vs_ncobras():
    query_budget = 200
    test_names = [
        "new_cobras",
        "new_ncobras",
        # "new_ncobras_with_!=1",
        # "new_ncobras_no_cached_cycles",
        # "new_ncobras_bug_fixed"

    ]
    noise_percentages = [-1, 0.05, 0.10]
    for noise_percentage in noise_percentages:
        noise_text = str(noise_percentage) if noise_percentage != -1 else "no"
        compare_names = []
        for test_name in test_names:
            compare_names.append(f"{test_name}_{noise_text}_budget_{query_budget}")
        # compare_names.append("NCOBRASplus_0.05_noise_budget200_pnoise0.05_threshold0.95")
        calculate_aris_and_compare_for_tests(f"cobras_vs_ncobras_new_noise_{noise_text}", compare_names, nb_of_cores=4,
                                             query_budget=200)


def test_varying_amounts_of_noise_NPU_COSC():
    from generate_clusterings.algorithms.my_cosc import MyCOSCMatlab
    from generate_clusterings.algorithms.my_npu import NPU
    test_collection = []
    clusterer = NPU(MyCOSCMatlab(run_fast_version=True), debug=True)
    # dataset_names = ['hepatitis']
    dataset_names = Dataset.get_standard_dataset_names()
    dataset_names.remove("hepatitis")
    nb_of_runs = 3
    test_collection.extend(
        test_varying_amounts_of_noise("new_NPU_COSC_fast", clusterer, cosc_result_extractor, dataset_names=dataset_names,
                                      query_budget=200, nb_of_runs=nb_of_runs))
    execute_list_of_clustering_tasks(test_collection, tests_per_batch=200)


def test_varying_amounts_of_noise_cobras():
    test_collection = []
    # plain COBRAS
    clusterer = COBRAS(correct_noise=False)
    dataset_names = Dataset.get_standard_dataset_names()
    nb_of_runs = 3
    test_collection.extend(
        test_varying_amounts_of_noise("new_cobras", clusterer, cobras_result_extractor, dataset_names=dataset_names,
                                      query_budget=200, nb_of_runs=nb_of_runs))
    execute_list_of_clustering_tasks(test_collection, tests_per_batch=200)


def test_varying_amounts_of_noise_ncobras():
    test_collection = []
    dataset_names = Dataset.get_standard_dataset_names()
    nb_of_runs = 3

    # nCOBRAS
    clusterers = [COBRAS(noise_probability=0.05, minimum_approximation_order=3, maximum_approximation_order=8,
                         certainty_threshold=0.95),
                  COBRAS(noise_probability=0.05, minimum_approximation_order=3, maximum_approximation_order=8,
                         certainty_threshold=0.95),
                  COBRAS(noise_probability=0.10, minimum_approximation_order=3, maximum_approximation_order=8,
                         certainty_threshold=0.95)]
    test_collection.extend(
        test_varying_amounts_of_noise("new_ncobras", clusterers, cobras_result_extractor, dataset_names=dataset_names,
                                      query_budget=200, nb_of_runs=nb_of_runs))

    execute_list_of_clustering_tasks(test_collection, tests_per_batch=200)
    # run_clustering_tasks_locally(test_collection)


def test_runtimes_ncobras():
    clusterer = COBRAS(noise_probability=0.05, minimum_approximation_order=3, maximum_approximation_order=10,
                       certainty_threshold=0.95)
    querier = ProbabilisticNoisyQuerier(None, None, 0.05, 100)
    tasks = make_n_run_10_fold_cross_validation("ncobras_0.05noise_runtimes", clusterer, querier,
                                                Dataset.get_standard_dataset_names(), 3, cobras_result_extractor)
    execute_list_of_clustering_tasks(tasks, tests_per_batch=200)


def test_varying_amounts_of_noise(test_name_raw, clusterers, result_extractor, query_budget=200, nb_of_runs=3,
                                  noise_percentages=(-1, 0.05, 0.1),
                                  dataset_names=Dataset.get_standard_dataset_names()):
    if not isinstance(clusterers, list):
        clusterers = [clusterers for _ in range(3)]
    assert len(clusterers) == len(noise_percentages)
    test_collection = []
    for clusterer, noise_percentage in zip(clusterers, noise_percentages):
        noise_text = str(noise_percentage) if noise_percentage != -1 else "no"
        test_name = f"{test_name_raw}_{noise_text}_budget_{query_budget}"
        querier = ProbabilisticNoisyQuerier(None, None, noise_percentage, query_budget)
        additional_tasks = make_n_run_10_fold_cross_validation(test_name, clusterer, querier, dataset_names, nb_of_runs,
                                                               result_extractor)
        test_collection.extend(additional_tasks)
    return test_collection


if __name__ == '__main__':
    # test_collection = test_varying_amounts_of_noise("TEST", COBRAS(correct_noise=False), None, 10, 1)
    # execute_list_of_clustering_tasks(test_collection)
    # execute_list_of_clustering_tasks([ClusteringTask(COBRAS(), "iris", None, None, ProbabilisticNoisyQuerier(None, None, 0.05,10),RESULTS_PATH+"/test_cobras.txt")])
    # test_varying_amounts_of_noise_all_algorithms()
    test_varying_amounts_of_noise_NPU_COSC()
    compare_clustering_quality_all_varying_amounts_of_noise()
    # compare_clustering_quality_all_varying_amounts_of_noise_OLD()
    # compare_runtimes()
    # compare_cobras_vs_ncobras()
