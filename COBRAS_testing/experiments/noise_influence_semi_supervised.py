import json
from pathlib import Path

from tqdm import tqdm

from config import RESULTS_PATH,CONSTRAINTS_PATH
from datasets import Dataset
from before_clustering.generate_constraints import generate_correct_random_constraint_set, add_noise_to_constraint_set, \
    generate_correct_min_max_active_constraint_set
from evaluate_clusterings.calculate_precision_recall_semi_supervised import collect_semi_supervised_results
from generate_clusterings.algorithms.my_mpckmeans_java import MyMPCKMeansJava
from generate_clusterings.semi_supervised_run_functions import run_semi_supervised_algo
from run_with_dask.run_with_dask import delayed_dataset_submission_function, \
    execute_list_of_futures_with_dataset_requirements

def tuple_list(cons):
    return [(i1,i2) for i1, i2 in cons]


def check_influence_of_noise_on_MPCK_means():
    random_name = "random_constraints_200"
    active_name = "active_constraints_200"
    active_noisy = "active_noisy_constraints_200"
    random_noisy = "random_noisy_constraints_200"
    datasets = list(Dataset.datasets())
    tests = run_mpck_means_with_constraints("MPCKJava_random_w=100", random_name,  datasets, 10, 100)
    tests.extend(run_mpck_means_with_constraints("MPCKJava_active_w=10000", active_name, datasets, 10, 10000))
    tests.extend(run_mpck_means_with_constraints("MPCKJava_noisy_random_w=100", random_noisy, datasets, 10, 100))
    tests.extend(run_mpck_means_with_constraints("MPCKJava_noisy_active_w=100", active_noisy, datasets, 10,100))
    execute_list_of_futures_with_dataset_requirements(tests, 100)
    random = collect_semi_supervised_results("MPCKJava_random")
    noisy_random = collect_semi_supervised_results("MPCKJava_noisy_random")
    # print("RANDOM")
    # print(random.sort_index())
    # print("RANDOM W=100")
    # print(collect_semi_supervised_results("MPCKJava_random_w=100").sort_index())
    # print("NOISY RANDOM")
    # print(noisy_random.sort_index())
    print("ACTIVE")
    print(collect_semi_supervised_results("MPCKJava_active").sort_index())
    print("ACTIVE w=10000")
    print(collect_semi_supervised_results("MPCKJava_active_w=10000").sort_index())
    # print("ACTIVE NOISY")
    # print(collect_semi_supervised_results("MPCKJava_noisy_active").sort_index())







def run_mpck_means_with_constraints(test_name, constraint_set_name, datasets,  nb_of_runs_per_dataset, w):
    tests_with_dataset_requirement = []
    for dataset in tqdm(datasets):
        result_path = Path(RESULTS_PATH) /"semi_supervised_results" / test_name / dataset.name
        constraint_set_path = Path(CONSTRAINTS_PATH)/constraint_set_name / dataset.name
        result_path.mkdir(parents=True, exist_ok=True)
        for i in range(nb_of_runs_per_dataset):
            full_result_path = result_path / f"run{i}.txt"
            if full_result_path.exists():
                continue
            with (constraint_set_path / f"constraints_{i}.json").open(mode = 'r') as input_file:
                ml, cl, noisy_ml, noisy_cl = json.load(input_file)
            ml, cl, noisy_ml, noisy_cl = map(tuple_list, [ml,cl,noisy_ml, noisy_cl])
            clusterer = MyMPCKMeansJava(w=w, take_transitive_closure=False)

            tests_with_dataset_requirement.append(
                    delayed_dataset_submission_function(run_semi_supervised_algo, dataset.name, clusterer, ml,
                                                        cl, noisy_ml, noisy_cl, full_result_path))
    return tests_with_dataset_requirement



if __name__ == '__main__':
    # debug_test()
    check_influence_of_noise_on_MPCK_means()
