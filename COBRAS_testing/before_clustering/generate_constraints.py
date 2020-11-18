import itertools
import json
import random
from pathlib import Path

import numpy as np
from active_semi_clustering.active.pairwise_constraints import  ExampleOracle

from before_clustering.explore_consolidate import ExploreConsolidate
from config import CONSTRAINTS_PATH
from datasets import Dataset


def list_of_tuples(array):
    return [(int(i1), int(i2)) for i1, i2 in array]


def generate_correct_min_max_active_constraint_set(dataset, nb_of_constraints):
    active_querier = ExploreConsolidate(dataset.number_of_classes())
    oracle = ExampleOracle(dataset.target, nb_of_constraints)
    active_querier.fit(dataset.data, oracle)
    ml, cl = active_querier.pairwise_constraints_
    return list_of_tuples(ml), list_of_tuples(cl)


def generate_correct_random_constraint_set(dataset, nb_of_constraints, nb_of_instances=None):
    if nb_of_instances is None:
        nb_of_instances = dataset.number_of_instances()
    instances_to_use = random.sample(list(range(dataset.number_of_instances())), nb_of_instances)
    all_possible_constraints = list(itertools.combinations(instances_to_use, 2))
    constraints = random.sample(all_possible_constraints, nb_of_constraints)
    ml, cl = [], []
    for i1, i2 in constraints:
        if dataset.target[i1] == dataset.target[i2]:
            ml.append((i1, i2))
        else:
            cl.append((i1, i2))
    return ml, cl


def add_noise_to_constraint_set(correct_ml, correct_cl, nb_noisy_constraints):
    correct_ml, correct_cl = np.array(correct_ml), np.array(correct_cl)
    nb_of_constraints = len(correct_ml) + len(correct_cl)
    noisy_indices = random.sample(list(range(nb_of_constraints)), k=nb_noisy_constraints)
    ml_noisy_indices = np.array([idx for idx in noisy_indices if idx < len(correct_ml)])
    cl_noisy_indices = np.array([idx - len(correct_ml) for idx in noisy_indices if idx >= len(correct_ml)])

    # cl constraints that are actually ml
    noisy_cl = correct_ml[ml_noisy_indices] if len(ml_noisy_indices) > 0 else []
    # ml constraints that are actually cl
    noisy_ml = correct_cl[cl_noisy_indices] if len(cl_noisy_indices) > 0 else []

    new_correct_ml = np.delete(correct_ml, ml_noisy_indices, axis=0) if len(ml_noisy_indices) > 0 else correct_ml
    new_correct_cl = np.delete(correct_cl, cl_noisy_indices, axis=0) if len(cl_noisy_indices) > 0 else correct_cl

    ml = np.concatenate((new_correct_ml, noisy_ml)).tolist() if len(noisy_ml) != 0 else new_correct_ml.tolist()
    cl = np.concatenate((new_correct_cl, noisy_cl)).tolist() if len(noisy_cl) != 0 else new_correct_cl.tolist()
    return list_of_tuples(ml), list_of_tuples(cl), list_of_tuples(noisy_ml), list_of_tuples(noisy_cl)


def generate_noisy_constraint_from_constraints(correct_constraint_set_name, result_constraint_set_name,
                                               nb_of_noisy_constraints):
    correct_constraint_path = Path(CONSTRAINTS_PATH) / correct_constraint_set_name
    noisy_constraint_path = Path(CONSTRAINTS_PATH) / result_constraint_set_name
    for dataset_directory in correct_constraint_path.iterdir():
        if dataset_directory.is_dir():
            for run_file in dataset_directory.iterdir():
                result_file = noisy_constraint_path / dataset_directory.name / run_file.name
                if result_file.exists():
                    continue
                with run_file.open(mode='r') as input_file:
                    ml, cl, _, _ = json.load(input_file)
                noise_ml, noise_cl, noisy_ml, noisy_cl = add_noise_to_constraint_set(ml, cl, nb_of_noisy_constraints)

                result_file.parent.mkdir(parents=True, exist_ok=True)
                with result_file.open(mode='w') as output_file:
                    json.dump((noise_ml, noise_cl, noisy_ml, noisy_cl), output_file)


def generate_random_constraints(datasets, nb_of_constraints, number_of_runs):
    generate_constraints_from_generator(datasets, generate_correct_random_constraint_set,
                                        f"random_constraints_{nb_of_constraints}", nb_of_constraints, number_of_runs)


def generate_active_constraints(datasets, nb_of_constraints, number_of_runs):
    generate_constraints_from_generator(datasets, generate_correct_min_max_active_constraint_set,
                                        f"active_constraints_{nb_of_constraints}", nb_of_constraints, number_of_runs)


def generate_constraints_from_generator(datasets, generator, constraint_set_name, number_of_constraints,
                                        number_of_runs):
    path = Path(CONSTRAINTS_PATH) / constraint_set_name
    for dataset, run_index in itertools.product(datasets, range(number_of_runs)):
        file_path = path / dataset.name / f"constraints_{run_index}.json"
        if file_path.exists():
            continue
        ml, cl = generator(dataset, number_of_constraints)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open(mode='w') as output_file:
            json.dump((ml, cl, [], []), output_file)

if __name__ == '__main__':
    datasets = list(Dataset.datasets(preprocessed=True))
    nb_of_constraints = 200
    nb_of_runs = 10
    nb_of_noise = 20
    generate_random_constraints(datasets, nb_of_constraints,nb_of_runs)
    print("generating active")
    generate_active_constraints(datasets, nb_of_constraints, nb_of_runs)
    generate_noisy_constraint_from_constraints(f"random_constraints_{nb_of_constraints}", f"random_noisy_constraints_{nb_of_constraints}", nb_of_noise)
    generate_noisy_constraint_from_constraints(f"active_constraints_{nb_of_constraints}",
                                               f"active_noisy_constraints_{nb_of_constraints}", nb_of_noise)

