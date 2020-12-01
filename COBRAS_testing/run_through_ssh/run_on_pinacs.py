from datasets import Dataset
from evaluate_clusterings.calculate_average_runtimes import calculate_average_runtimes_for_tests
from evaluate_clusterings.calculate_aligned_rank import calculate_and_write_aligned_rank
from evaluate_clusterings.calculate_aris import calculate_n_times_n_fold_aris_for_testnames_OLD, \
    calculate_coprf_like_aris_for_testnames
from evaluate_clusterings.calculate_average_aris import calculate_average_aris, calculate_average_aris_cop_rf
from present_results.plot_aligned_rank import plot_rank_comparison_file
from present_results.plot_aris import plot_average_ARI_per_dataset, plot_overall_average_ARI
from present_results.plot_runtimes import plot_runtime_comparison
from run_through_ssh.generate_task_file import TestCollection, cobras_algorithm_settings_to_string, \
    probabilistic_noisy_querier_settings_to_string, mpck_means_algorithm_settings_to_string

from run_through_ssh.run_task_file import run_task_file_locally
from run_through_ssh.run_task_file_over_ssh import run_task_files_over_ssh, killall_on_machines


def himecs_generate_computer_info(start_index=1, nb_of_machines=2):
    """ generates computer info for the himec machines """
    l = [("himec01", 12), ("himec02", 12), ("himec03", 16), ("himec04", 16)]
    return l[start_index:start_index + nb_of_machines]


def generate_computer_info(start_index=0, nb_of_machines=5):
    """
        generates computer info for the normal pinac machines
    :param start_index:
    :param nb_of_machines:
    :return:
    """
    machines = []
    machines.extend([("pinac" + str(i), 4) for i in range(21, 31)])

    machines.extend([("pinac" + str(i), 4) for i in range(31, 41)])
    machines.extend([("pinac-" + i, 4) for i in "abcd"])
    if start_index + nb_of_machines > len(machines):
        raise Exception("too much machines asked")
    return machines[start_index: start_index + nb_of_machines]


def run_tests_local(tests, nb_of_cores=3):
    """ runs the given test_cobras collection locally"""
    task_file_name = "local_task.txt"
    tests.divide_over_task_files([task_file_name])
    run_task_file_locally(task_file_name, nb_of_cores)


def run_tests_over_SSH_on_machines(tests, machine_info):
    """ runs the given tests over ssh """
    nb_of_computers = len(machine_info)
    task_file_names = ["machine" + str(i) + ".txt" for i in range(nb_of_computers)]
    print("dividing {} tasks over {} task files".format(len(tests.test_list), nb_of_computers))
    tests.divide_over_task_files(task_file_names)
    print("running task files through ssh on {} machines".format(nb_of_computers))
    run_task_files_over_ssh(task_file_names, machine_info, tests.resulting_file_names, new_machine=True, skip_validation=SKIP_VALIDATION)



def calculate_aris_and_compare_for_tests(comparison_name, test_names, line_names=None, nb_of_cores=8, recalculate=False,
                                         offsets_dict = None, query_budget=None):
    calculate_n_times_n_fold_aris_for_testnames_OLD(test_names, nb_cores=nb_of_cores, recalculate=recalculate)
    calculate_average_aris(test_names, query_budget)
    calculate_average_runtimes_for_tests(test_names, query_budget)
    compare_algorithms_and_plot_results_n_times_n_fold(comparison_name, test_names, line_names, offset_dict=offsets_dict)


def compare_algorithms_and_plot_results_n_times_n_fold(comparison_name, algorithms, line_names=None, offset_dict = None,):
    if line_names is None:
        line_names = algorithms
    print("calculating rank comparison file")
    calculate_and_write_aligned_rank(algorithms, comparison_name)
    print("plotting ARI comparisons")
    # plot_average_ARI_per_dataset(comparison_name, algorithms, line_names)
    plot_overall_average_ARI(comparison_name, algorithms, line_names, offset_dict=offset_dict)
    standard_colors =[u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
    colors_to_use = [u'#2ca02c',u'#ff7f0e', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
    plot_runtime_comparison(comparison_name,algorithms, line_names, colors_to_use)
    print("plotting aligned rank comparison")
    plot_rank_comparison_file(comparison_name, algorithms, line_names)




###################################################
####       example tests for COBRAS ###############
###################################################

def simple_cobras_tests():
    # this class contains the logic to build different kind of test_cobras cases
    # if you want another testing scenario you can add it to this class
    tests = TestCollection()
    tests.add_10_times_10_fold_test("<TEST NAME>", "COBRAS",
                                    cobras_algorithm_settings_to_string(0.10, 3, 7, 0.91, 0.91, False, False),
                                    Dataset.get_dataset_names(),
                                    "probability_noise_querier",  # name of the querier
                                    probabilistic_noisy_querier_settings_to_string(0,
                                                                                   200))  # noise probability 0 --> no noise
    # this runs the tests locally over the number of cores specified
    run_tests_local(tests, nb_of_cores=4)

    # after running several of the above you can compare different results as follows
    comparison_name = "NAME OF THE COMPARISON"
    test_names = ["<TEST NAME>",
                  "<OTHER TEST NAME>"]  # these should be the same string as the first argument of tests.add_10_times_10_fold_test
    line_names = ["<simple name for <TEST NAME>>",
                  "<OTHER SIMPLE NAME>"]  # these names are displayed in the legend of the plots instead of test_cobras names (test_cobras names should be unique and can thus become very large)
    # this will calculate all the aris and compare the tests this is not possible over SSH but this is not as much works as well
    calculate_aris_and_compare_for_tests(comparison_name, test_names, line_names, query_budget=200, nb_of_cores=4)


###################################################
#####             COP-RF like tests ###############
###################################################

def cobras_cop_rf_varying_noise_test():
    print("making tests")

    tests = TestCollection()
    datasets = ["ionosphere", "iris", "segmentation_training_only", "parkinsons", "glass"]
    constraint_percentage = 0.003
    noise_percentages = [0.05, 0.1, 0.15, 0.20, 0.25, 0.30]

    tests.add_cop_rf_like_tests_for_cobras_varying_noise("ncobras_cop_rf_like2", datasets, constraint_percentage,
                                                         noise_percentages, min_approx_order=3, max_approx_order=7)
    run_tests_over_SSH_on_machines(tests, generate_computer_info(nb_of_machines=5))
    test_names = ["ncobras_cop_rf_like2"]
    calculate_coprf_like_aris_for_testnames(test_names)
    calculate_average_aris_cop_rf(test_names)


##################################################
####      varying amounts of noise tests##########
##################################################

def NPU_MPCKMeans_VaryingAmountsOfNoise():
    print("making tests")
    tests = TestCollection()
    query_budget = 200
    for noise_percentage in [-1, 0.05, 0.10, 0.20]:
        noise_text = str(noise_percentage) if noise_percentage != -1 else "no"
        tests.add_10_times_10_fold_test("NPU_MPCKmeans_{}_noise_budget{}".format(noise_text, query_budget),
                                        "NPU_MPCKmeans",
                                        mpck_means_algorithm_settings_to_string(),
                                        Dataset.get_standard_dataset_names(),
                                        "probability_noise_querier",
                                        probabilistic_noisy_querier_settings_to_string(noise_percentage, query_budget))
    run_tests_over_SSH_on_machines(tests, MACHINES_TO_USE)
    comparison_name = "NPU_MPCKmeans_varying_amounts_of_noise"
    test_names = ["NPU_MPCKmeans_no_noise_budget200", "NPU_MPCKmeans_0.05_noise_budget200",
                  "NPU_MPCKmeans_0.10_noise_budget200", "NPU_MPCKmeans_0.20_noise_budget200"]
    line_names = ["NPU_MPCKmeans_no_noise", "NPU_MPCKmeans_0.05_noise", "NPU_MPCKmeans_0.10_noise",
                  "NPU_MPCKmeans_0.20_noise"]
    calculate_aris_and_compare_for_tests(comparison_name, test_names, line_names, query_budget=query_budget)


def NPU_Cosc_VaryingAmountsOfNoise():
    print("making tests")
    tests = TestCollection()
    query_budget = 100
    for noise_percentage in [-1, 0.10]:
        noise_text = str(noise_percentage) if noise_percentage != -1 else "no"
        tests.add_10_times_10_fold_test("NPU_Cosc_{}_noise_budget{}".format(noise_text, query_budget), "NPU_COSC",
                                        "no_parameters", Dataset.get_non_face_news_spam_names(),
                                        "probability_noise_querier",
                                        probabilistic_noisy_querier_settings_to_string(noise_percentage, query_budget))
    run_tests_over_SSH_on_machines(tests, MACHINES_TO_USE)
    comparison_name = "NPU_Cosc_varying_amounts_of_noise"
    test_names = ["NPU_Cosc_no_noise_budget100", "NPU_Cosc_0.1_noise_budget100"]
    line_names = None
    calculate_aris_and_compare_for_tests(comparison_name, test_names, line_names, query_budget=query_budget)


def cobras_VaryingAmountsOfNoise():
    print("making tests")
    tests = TestCollection()
    query_budget = 200
    for noise_percentage in [-1, 0.05, 0.10, 0.20]:
        noise_text = str(noise_percentage) if noise_percentage != -1 else "no"
        tests.add_10_times_10_fold_test("cobras_{}_noise_budget{}".format(noise_text, query_budget), "COBRAS",
                                        cobras_algorithm_settings_to_string(0.10, 3, 7, 0.91, 0.91, False, False),
                                        Dataset.get_dataset_names(), "probability_noise_querier",
                                        probabilistic_noisy_querier_settings_to_string(noise_percentage, query_budget))
    run_tests_over_SSH_on_machines(tests, MACHINES_TO_USE)
    comparison_name = "cobras_varying_amounts_of_noise"
    test_names = ["cobras_{}_noise_budget200".format(i) for i in ["no", 0.05, 0.10, 0.20]]
    line_names = None
    calculate_aris_and_compare_for_tests(comparison_name, test_names, line_names)

def ncobras_plus_runtime_test():
    tests = TestCollection()
    query_budget = 100
    noise_precentage = 0.05
    threshold = 0.95
    tests.add_10_times_10_fold_test("NCOBRASplus_{}_noise_budget{}_pnoise{}_threshold{}_runtimes".format(noise_precentage, query_budget, noise_precentage, threshold),
                                    "COBRAS",
                                    cobras_algorithm_settings_to_string(noise_precentage, min_approx_order=3, max_approx_order=10, keep_threshold=threshold, reuse_threshold=threshold, correct_noise=True, use_all_cycles=False),
                                    Dataset.get_dataset_names(), "probability_noise_querier", probabilistic_noisy_querier_settings_to_string(noise_precentage, query_budget), nb_of_runs=10)
    run_tests_over_SSH_on_machines(tests, MACHINES_TO_USE)

def ncobras_plus_runtime_comparison():
    noise_text = 0.05
    p_noise = 0.05
    ncobras_name = "NCOBRASplus_{}_noise_budget100_pnoise{}_threshold0.95_runtimes".format(noise_text, p_noise)
    cobras_noise = "cobras_{}_noise_budget200".format(noise_text)
    npu_cosc_noise = "NPU_Cosc_{}_noise_budget200".format(noise_text)
    npu_mpck_noise = "NPU_MPCKmeans_{}_noise_budget200".format(noise_text)
    test_names = [ncobras_name, cobras_noise, npu_cosc_noise, npu_mpck_noise]
    line_names = ["ncobras", "cobras", "npu_cosc", "npu_mpck"]
    calculate_aris_and_compare_for_tests("ncobras_plus_runtime", test_names, line_names,
                                         query_budget=100)


def ncobras_plus_varying_amounts_of_noise():
    print("making tests")
    tests = TestCollection()
    query_budget = 200
    for noise_percentage in [-1, 0.05, 0.10]:
        noise_text = str(noise_percentage) if noise_percentage != -1 else "no"
        threshold = 0.95
        noise_percentage_to_use = noise_percentage if noise_percentage > 0 else 0.10
        tests.add_10_times_10_fold_test(
            "NCOBRASplus_{}_noise_budget{}_pnoise{}_threshold{}".format(noise_text, query_budget, noise_percentage_to_use,
                                                                        threshold), "COBRAS",
            cobras_algorithm_settings_to_string(noise_percentage_to_use, 3, 10, threshold, threshold, True, False),
            Dataset.get_dataset_names(), "probability_noise_querier",
            probabilistic_noisy_querier_settings_to_string(noise_percentage, query_budget), nb_of_runs=10)
    run_tests_over_SSH_on_machines(tests, MACHINES_TO_USE)


def ncobras_plus_plot():
    comparison_name_template = "NCOBRASPlus_comparison_{}_noise"
    offset_dict_no = {"cobras": 0.01, "ncobras": -0.01}
    offset_dict_5 = {}
    offset_dict_10 = {"cobras": -0.015, "npu_cosc": 0.015}
    for noise_text,p_noise,offsets in [('0.05',0.05,offset_dict_5), ('0.1',0.10, offset_dict_10), ('no', 0.10, offset_dict_no)]:
        comparison_name = comparison_name_template.format(noise_text)
        ncobras_name = "NCOBRASplus_{}_noise_budget200_pnoise{}_threshold0.95".format(noise_text,p_noise)
        cobras_noise = "cobras_{}_noise_budget200".format(noise_text)
        npu_cosc_noise = "NPU_Cosc_{}_noise_budget200".format(noise_text)
        npu_mpck_noise = "NPU_MPCKmeans_{}_noise_budget200".format(noise_text)
        test_names = [cobras_noise,ncobras_name,  npu_cosc_noise, npu_mpck_noise]
        line_names = [ "cobras", "ncobras","npu_cosc", "npu_mpck"]
        calculate_aris_and_compare_for_tests(comparison_name,test_names, line_names, offsets_dict = offsets, query_budget=200)




def plot_varying_noise_comparison():
    test_names = [
        "cobras_noise",
        "cobras_no_noise",
        "NPU_MPCKmeans_no_noise",
        "NPU_MPCKmeans_0.10_noise",
        "NPU_Cosc_no_noise_budget100",
        "NPU_Cosc_0.1_noise_budget100",
        "cobras_0.10_p0.1_t0.99_noise_budget200"
    ]
    line_names = None
    comparison_name = "all_algorithms_noise_vs_no_noise"
    calculate_aris_and_compare_for_tests(comparison_name, test_names, line_names, query_budget=100)


###############################################
###     synthetic datasets                   ##
###############################################
def synthetic_datasets_comparison():
    print("making tests")
    tests = TestCollection()
    datasets = ["compound", "flame", "jain", "pathbased", "spiral"]
    algorithms = "COBRAS"
    tests.add_10_times_10_fold_test("COSC_synthetic_no_noise",
                                    "NPU_COSC",
                                    "no parameters",
                                    datasets,
                                    "probability_noise_querier",
                                    probabilistic_noisy_querier_settings_to_string(-1, 100),
                                    nb_of_runs=3)
    tests.add_10_times_10_fold_test("COSC_synthetic_0.10_noise",
                                    "NPU_COSC",
                                    "no parameters",
                                    datasets,
                                    "probability_noise_querier",
                                    probabilistic_noisy_querier_settings_to_string(0.10, 100),
                                    nb_of_runs=3)
    tests.add_10_times_10_fold_test("MPCK_means_synthetic_0.10_noise",
                                    "NPU_MPCKmeans",
                                    mpck_means_algorithm_settings_to_string(),
                                    datasets,
                                    "probability_noise_querier",
                                    probabilistic_noisy_querier_settings_to_string(0.10, 100),
                                    nb_of_runs=3)
    tests.add_10_times_10_fold_test("MPCK_means_synthetic_no_noise",
                                    "NPU_MPCKmeans",
                                    mpck_means_algorithm_settings_to_string(),
                                    datasets,
                                    "probability_noise_querier",
                                    probabilistic_noisy_querier_settings_to_string(-1, 100),
                                    nb_of_runs=3)
    tests.add_10_times_10_fold_test("nCOBRAS_synthetic_no_noise",
                                    "COBRAS",
                                    cobras_algorithm_settings_to_string(0.10, min_approx_order=3, max_approx_order=5,
                                                                        keep_threshold=0.99, reuse_threshold=0.99,
                                                                        correct_noise=False),
                                    datasets,
                                    "probability_noise_querier",
                                    probabilistic_noisy_querier_settings_to_string(-1, 100),
                                    nb_of_runs=3)
    tests.add_10_times_10_fold_test("nCOBRAS_synthetic_0.10_noise",
                                    "COBRAS",
                                    cobras_algorithm_settings_to_string(0.10, min_approx_order=3, max_approx_order=5,
                                                                        keep_threshold=0.99, reuse_threshold=0.99),
                                    datasets,
                                    "probability_noise_querier",
                                    probabilistic_noisy_querier_settings_to_string(0.10, 100),
                                    nb_of_runs=3)
    run_tests_over_SSH_on_machines(tests, himecs_generate_computer_info(2, 2))
    comparison_name = "MPCK_vs_cobras_synthetic"
    test_names = ["MPCK_means_synthetic_0.10_noise", "MPCK_means_synthetic_no_noise", "nCOBRAS_synthetic_0.10_noise",
                  "nCOBRAS_synthetic_no_noise"]  # , "COSC_synthetic_0.10_noise", "COSC_synthetic_no_noise"]
    line_names = None
    calculate_aris_and_compare_for_tests(comparison_name, test_names, line_names, nb_of_cores=24, query_budget=100)


################################################
##### cobras parameter studies               ###
################################################

def ncobras_noise_comparison_fixed_noise_changing_p_noise():
    print("making tests")
    tests = TestCollection()

    tests.add_10_times_10_fold_test("ncobras_0.10_noise_0.05_p_noise", "COBRAS",
                                    cobras_algorithm_settings_to_string(0.05, 3, 7, 0.96, 0.96, True, False),
                                    Dataset.get_dataset_names(), "probability_noise_querier",
                                    probabilistic_noisy_querier_settings_to_string(0.10, 250))
    tests.add_10_times_10_fold_test("ncobras_0.10_noise_0.10_p_noise", "COBRAS",
                                    cobras_algorithm_settings_to_string(0.10, 3, 7, 0.91, 0.91, True, False),
                                    Dataset.get_dataset_names(), "probability_noise_querier",
                                    probabilistic_noisy_querier_settings_to_string(0.10, 250))
    tests.add_10_times_10_fold_test("ncobras_0.10_noise_0.15_p_noise", "COBRAS",
                                    cobras_algorithm_settings_to_string(0.15, 3, 7, 0.91, 0.91, True, False),
                                    Dataset.get_dataset_names(), "probability_noise_querier",
                                    probabilistic_noisy_querier_settings_to_string(0.10, 250))
    tests.add_10_times_10_fold_test("ncobras_0.10_noise_0.20_p_noise", "COBRAS",
                                    cobras_algorithm_settings_to_string(0.20, 3, 7, 0.91, 0.91, True, False),
                                    Dataset.get_dataset_names(), "probability_noise_querier",
                                    probabilistic_noisy_querier_settings_to_string(0.10, 250))
    run_tests_over_SSH_on_machines(tests, generate_computer_info(start_index=21, nb_of_machines=5))
    comparison_name = "ncobras_parameter_sensitivity"
    test_names = [
        "ncobras_0.10_noise_0.05_p_noise",
        "ncobras_0.10_noise_0.10_p_noise",
        "ncobras_0.10_noise_0.15_p_noise",
        "ncobras_0.10_noise_0.20_p_noise"
    ]

    line_names = None
    calculate_aris_and_compare_for_tests(comparison_name, test_names, line_names)


def ncobras_noise_comparison_changing_noise():
    print("making tests")
    tests = TestCollection()
    tests.add_10_times_10_fold_test("ncobras_0.05_noise_0.05_p_noise", "COBRAS",
                                    cobras_algorithm_settings_to_string(0.05, 3, 7, 0.96, 0.96, True, False),
                                    Dataset.get_dataset_names(), "probability_noise_querier",
                                    probabilistic_noisy_querier_settings_to_string(0.05, 250))
    tests.add_10_times_10_fold_test("ncobras_0.10_noise_0.10_p_noise", "COBRAS",
                                    cobras_algorithm_settings_to_string(0.1, 3, 7, 0.91, 0.91, True, False),
                                    Dataset.get_dataset_names(), "probability_noise_querier",
                                    probabilistic_noisy_querier_settings_to_string(0.10, 250))
    tests.add_10_times_10_fold_test("ncobras_0.15_noise_0.15_p_noise", "COBRAS",
                                    cobras_algorithm_settings_to_string(0.15, 3, 7, 0.91, 0.91, True, False),
                                    Dataset.get_dataset_names(), "probability_noise_querier",
                                    probabilistic_noisy_querier_settings_to_string(0.15, 250))
    tests.add_10_times_10_fold_test("ncobras_0.20_noise_0.20_p_noise", "COBRAS",
                                    cobras_algorithm_settings_to_string(0.20, 3, 7, 0.91, 0.91, True, False),
                                    Dataset.get_dataset_names(), "probability_noise_querier",
                                    probabilistic_noisy_querier_settings_to_string(0.20, 250))
    run_tests_over_SSH_on_machines(tests, generate_computer_info(start_index=21, nb_of_machines=5))
    comparison_name = "ncobras_parameter_sensitivity"
    test_names = [
        "ncobras_0.05_noise_0.05_p_noise",
        "ncobras_0.10_noise_0.10_p_noise",
        "ncobras_0.15_noise_0.15_p_noise",
        "ncobras_0.20_noise_0.20_p_noise"
    ]

    line_names = None
    calculate_aris_and_compare_for_tests(comparison_name, test_names, line_names)


def cobras_parameter_comparison_plots():
    test_names = ["cobras_no_noise", "noise_robust_cobras", "NPU_MPCKmeans_no_noise", "NPU_MPCKmeans_0.10_noise",
                  "cobras_0.10_p0.1_t0.99_noise_budget200"]
    compare_algorithms_and_plot_results_n_times_n_fold("cobras_vs_mpck_means_bis", test_names)
    # all_ps = [0.05, 0.10,0.15,0.20]
    # all_ts = [0.99,0.96, 0.91, 0.86, 0.81]
    # test_name_template = "cobras_0.10_p{}_t{}_noise_budget100"
    # for t in all_ts:
    #     comparison_name = "different_ps_for_t{}".format(t)
    #     comparison_names = ["cobras_no_noise"]
    #     for p in all_ps:
    #         comparison_names.append(test_name_template.format(p,t))
    #     compare_algorithms_and_plot_results_n_times_n_fold(comparison_name, comparison_names)
    #
    # for p in all_ps:
    #     comparison_name = "different_ts_for_p{}".format(p)
    #     comparison_names = ["cobras_no_noise"]
    #     for t in all_ts:
    #         comparison_names.append(test_name_template.format(p, t))
    #     compare_algorithms_and_plot_results_n_times_n_fold(comparison_name, comparison_names)


def cobras_parameter_comparison():
    print("making tests")
    query_budget = 200

    tests = TestCollection()
    test_names = ["cobras_no_noise"]
    test_dict = {
        0.05: [0.96, 0.99],
        0.10: [0.96, 0.99],
        0.15: [0.91, 0.96, 0.99],
        0.20: [0.86, 0.91, 0.96]
    }
    for p in [0.10, 0.15, 0.20]:
        t_values = test_dict[p]
        for t in t_values:
            test_names.append("cobras_0.10_p{}_t{}_noise_budget{}".format(p, t, query_budget))
            tests.add_10_times_10_fold_test("cobras_0.10_p{}_t{}_noise_budget{}".format(p, t, query_budget), "COBRAS",
                                            cobras_algorithm_settings_to_string(p, 3, 7, t, t, True, False),
                                            Dataset.get_non_face_news_spam_names(), "probability_noise_querier",
                                            probabilistic_noisy_querier_settings_to_string(0.10, query_budget),
                                            nb_of_runs=1)

    run_tests_over_SSH_on_machines(tests, himecs_generate_computer_info(start_index=3, nb_of_machines=2))
    comparison_name = "all_parameter_study"
    line_names = [test_name[12:-16] for test_name in test_names]
    calculate_aris_and_compare_for_tests(comparison_name, test_names, line_names, nb_of_cores=24, query_budget=200,
                                         recalculate=False)


################################################
######## tests to include or not?    ###########
################################################

def cobras_minimal_vs_all_cycles_test():
    print("making tests")

    tests = TestCollection()
    tests.add_10_times_10_fold_test("ncobras_minimal_cycles", "COBRAS",
                                    cobras_algorithm_settings_to_string(0.10, 3, 7, 0.91, 0.91, True, False),
                                    Dataset.get_standard_dataset_names(), "probability_noise_querier",
                                    probabilistic_noisy_querier_settings_to_string(0.10, 25))
    tests.add_10_times_10_fold_test("ncobras_all_cycles", "COBRAS",
                                    cobras_algorithm_settings_to_string(0.10, 3, 7, 0.91, 0.91, True, True),
                                    Dataset.get_standard_dataset_names(), "probability_noise_querier",
                                    probabilistic_noisy_querier_settings_to_string(0.10, 25))
    run_tests_over_SSH_on_machines(tests, nb_of_computers=5)
    comparison_name = "ncobras_minimal_vs_all_cycles"
    test_names = ["ncobras_minimal_cycles", "ncobras_all_cycles"]
    line_names = None
    calculate_aris_and_compare_for_tests(comparison_name, test_names, line_names)


#################################################
##### experiments tests                    ######
#################################################


MACHINES_TO_USE = generate_computer_info(start_index=11, nb_of_machines=3)+generate_computer_info(15,5)# + himecs_generate_computer_info(3,1)
SKIP_VALIDATION = True
if __name__ == '__main__':
    # killall_on_machines(MACHINES_TO_USE, new_machine=True)
    ### basic varying amounts of noise
    # NPU_Cosc_VaryingAmountsOfNoise()
    # NPU_MPCKMeans_VaryingAmountsOfNoise()
    # cobras_VaryingAmountsOfNoise()

    ### plots varying amounts of noise
    # plot_varying_noise_comparison()

    # synthetic_datasets_comparison()

    # parameter study plotting
    # plot_parameter_comparison_extreme_cases()
    # cobras_parameter_comparison()

    # ncobras_plus_varying_amounts_of_noise()
    # ncobras_plus_runtime_test()
    # ncobras_plus_runtime_comparison()
    ncobras_plus_plot()
    # cobras_minimal_vs_all_cycles_test()
    # cobras_cop_rf_varying_noise_test()

    # NPU_Cosc_VaryingAmountsOfNoise()
    # NPU_MPCKMeans_VaryingAmountsOfNoise()
    # cobras_parameter_comparison()
    # cobras_parameter_comparison_plots()
    # compare_MPCK_means_and_COBRAS()
    # cobras_test_again()
