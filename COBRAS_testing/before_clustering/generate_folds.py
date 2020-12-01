from pathlib import Path

from sklearn.model_selection import StratifiedKFold, KFold
import os
from config import FOLD_PATH
import numpy as np
import json

from datasets import Dataset

def get_train_indices_for_dataset(dataset_name, run_idx, fold_idx):
    fold_path = Path(FOLD_PATH, dataset_name, f"run{run_idx}_fold{fold_idx}.txt")
    with fold_path.open(mode="r") as input_file:
        result_dict = json.load(input_file)
    train_indices = result_dict['train_indices']
    return train_indices

def generate_folds_for_dataset():
    dataset_names = Dataset.get_dataset_names() + Dataset.interesting_2d_datasets()

    for dataset_name in dataset_names:

        dataset = Dataset(dataset_name)
        print("making folds for dataset ", dataset_name)
        os.makedirs(os.path.join(FOLD_PATH, dataset_name), exist_ok=True)
        for run_nb in range(10):
            # toon's code
            # skf = cross_validation.StratifiedKFold(labels, n_folds=5, shuffle=True)
            skf = StratifiedKFold(n_splits = 10, shuffle = True)
            # skf = KFold(n_splits=10, shuffle=True)
            labels = dataset.target

            for fold_nb, (train_indices, test_indices) in enumerate(skf.split(np.zeros(len(labels)), labels)):

                to_write = dict()
                to_write["train_indices"] = train_indices.tolist()
                to_write["test_indices"] = test_indices.tolist()
                if os.path.isfile(os.path.join(FOLD_PATH, dataset_name, "run{}_fold{}.txt".format(run_nb, fold_nb))):
                    print("fold file already exists! not overwriting!")
                    continue
                with open(os.path.join(FOLD_PATH, dataset_name, "run{}_fold{}.txt".format(run_nb, fold_nb)), mode = 'w') as fold_file:
                    json.dump(to_write, fold_file)

if __name__ == '__main__':
    generate_folds_for_dataset()