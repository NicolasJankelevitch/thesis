from sklearn.model_selection import StratifiedKFold
from util.datasets import Dataset
from util.config import FOLDS_PATH
import os, json
import numpy as np

"""
    This function can be used to generate stratified folds for testing different datasets.
    Needs FOLDS_PATH defined in config, it is where the folds will be saved.
"""


def generate_fold_for_datasets():
    nb_folds = 10
    # Check that we really want to override the current folds
    if os.path.isdir(os.path.join(FOLDS_PATH)):
        answ = input("Folds already exist, overwrite them? y/n: ")
        if answ != 'y':
            print("Folds creation aborted.")
            return

    dataset_names = Dataset.get_dataset_names() + Dataset.interesting_2d_datasets()
    for dataset_name in dataset_names:
        print("Creating folds for dataset ", dataset_name)
        dataset = Dataset(dataset_name)
        os.makedirs(os.path.join(FOLDS_PATH, dataset_name), exist_ok=True)

        for run_nb in range(nb_folds):
            skf = StratifiedKFold(n_splits=nb_folds, shuffle=True)
            labels = dataset.target

            for fold_nb, (train_indices, test_indices) in enumerate(skf.split(np.zeros(len(labels)), labels)):
                to_write = dict()
                to_write["train_indices"] = train_indices.tolist()
                to_write["test_indices"] = test_indices.tolist()
                with open(os.path.join(FOLDS_PATH, dataset_name, "run{}_fold{}.txt".format(run_nb, fold_nb)), mode='w') as fold_file:
                    json.dump(to_write, fold_file)


if __name__ == "__main__":
    generate_fold_for_datasets()