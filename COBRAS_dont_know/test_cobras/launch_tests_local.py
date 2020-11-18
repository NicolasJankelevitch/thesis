from test_cobras.evaluate_clustering.calculate_aligned_rank import calculate_and_write_aligned_rank
from test_cobras.evaluate_clustering.calculate_aris import calculate_n_times_n_fold_aris_for_testnames
from test_cobras.evaluate_clustering.calculate_average_aris import calculate_average_aris
from test_cobras.general_run_task import run_task_file
from test_cobras.present_results.plot_aligned_rank import plot_rank_comparison_file
from test_cobras.present_results.plot_aris import plot_average_ARI_per_dataset, plot_overall_average_ARI
from test_cobras.test_collection import TestCollection, cobras_algorithm_settings_to_string, weak_querier_settings_to_string
from util.datasets import Dataset

def baseline_cobras():
    n_queries = 100
    datasets = Dataset.interesting_2d_datasets() + Dataset.get_quick_dataset_names()
    tests = TestCollection()
    tests.add_10_times_10_fold_test("handle_low_nocap",
                                    "COBRAS",
                                    cobras_algorithm_settings_to_string(),
                                    datasets,
                                    "weak_querier",
                                    weak_querier_settings_to_string('local_nondet', max_prob=1, max_queries=n_queries))
    run_tests_local(tests, nb_of_cores=4)

    comparison_name = "final Cobras cap"
    test_names = ["cobras_no_uncertainty_s3", "cobras_no_uncertainty"]
    line_names = ["COBRAS: Max 8 SIs", "COBRAS: No max"]
    calculate_aris_and_compare_for_tests(comparison_name, test_names, line_names, query_budget=n_queries, nb_of_cores=6)


"""
    Do NOT modify
"""
def run_tests_local(tests, nb_of_cores=4):
    """ runs the given test_cobras collection locally"""
    task_file_name = "local_task.txt"
    tests.divide_over_task_files([task_file_name])
    run_task_file(task_file_name, nb_of_cores)


"""
    Do NOT modify
"""
def calculate_aris_and_compare_for_tests(comparison_name, test_names, line_names=None, nb_of_cores=8, recalculate=False, query_budget=250):
    calculate_n_times_n_fold_aris_for_testnames(test_names, nb_cores=nb_of_cores, recalculate=recalculate)
    calculate_average_aris(test_names, query_budget)
    compare_algorithms_and_plot_results_n_times_n_fold(comparison_name, test_names, line_names)


"""
    Do NOT modify
"""
def compare_algorithms_and_plot_results_n_times_n_fold(comparison_name, algorithms, line_names=None):
    if line_names is None:
        line_names = algorithms
    print("calculating rank comparison file")
    calculate_and_write_aligned_rank(algorithms, comparison_name)
    print("plotting ARI comparisons")
    plot_average_ARI_per_dataset(comparison_name, algorithms, line_names)
    plot_overall_average_ARI(comparison_name, algorithms, line_names)
    print("plotting aligned rank comparison")
    plot_rank_comparison_file(comparison_name, algorithms, line_names)


if __name__=="__main__":
    ssac()
    pass

