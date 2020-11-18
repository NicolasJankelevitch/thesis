from pathlib import Path

import jsonpickle
import numpy as np
from config import RESULTS_PATH
import pandas as pd


def collect_precision_and_recall_for_dataset(dataset_directory):
    precisions = []
    recalls = []

    for result_file in dataset_directory.iterdir():
        try:
            with result_file.open(mode='r') as input_file:
                string = input_file.readline()
                result_dict = jsonpickle.decode(string)
            precision = result_dict['precision']
            recall = result_dict['recall']
            if precision is None or recall is None:
                continue
            precisions.append(precision)
            recalls.append(recall)
        except Exception as e:
            print("error during processing of file:", result_file)
            print(e)
    if len(precisions) == 0 and len(recalls) == 0:
        return 0, 0
    return sum(precisions)/len(precisions), sum(recalls)/len(recalls)

def collect_false_positives_data(dataset_directory):
    data = []
    for result_file in dataset_directory.iterdir():
        try:
            with result_file.open(mode='r') as input_file:
                string = input_file.readline()
                result_dict = jsonpickle.decode(string)
            f_pos = result_dict["false_positives"]
            f_neg = result_dict["false_negatives"]
            t_pos = result_dict["true_positives"]
            t_neg = result_dict["true_negatives"]
            ari = result_dict["ari"]
            data.append((ari, len(t_pos), len(t_neg), len(f_pos), len(f_neg)))
        except Exception as e:
            print("error during processing of file:", result_file)
            print(e)
    data = np.array(data)
    return np.mean(data, axis=0)



def collect_semi_supervised_results(test_name):
    results_directory = Path(RESULTS_PATH) / "semi_supervised_results" / test_name
    data = []
    for dataset_directory in results_directory.iterdir():
        if dataset_directory.is_dir():
            mean_ari, mean_t_pos, mean_t_neg, mean_f_pos, mean_f_neg = collect_false_positives_data(dataset_directory)
            data.append((dataset_directory.name, mean_ari, mean_t_pos, mean_t_neg, mean_f_pos, mean_f_neg))
    df = pd.DataFrame(data, columns=['dataset', 'mean ari', 'mean true pos', 'mean true neg', 'mean false pos', 'mean false neg'])
    df.set_index('dataset', inplace=True)
    return df
