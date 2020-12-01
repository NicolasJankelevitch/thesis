import numpy as np
from tqdm import tqdm
from util.config import FOLDS_RESULT_PATH
import os
import json


class AverageARITask:
    def __init__(self,test_name, query_budget):
        self.test_name = test_name
        self.query_budget = query_budget

    def get_aris_dir(self):
        return os.path.join(FOLDS_RESULT_PATH, self.test_name, "aris")

    def get_average_ari_file(self):
        return os.path.join(FOLDS_RESULT_PATH, self.test_name, "ari_averages.txt")

    def read_all_ARIs(self, dataset_name):
        dataset_path = os.path.join(self.get_aris_dir(), dataset_name)
        all_aris = []
        for filename in os.listdir(dataset_path):
            with open(os.path.join(dataset_path,filename), mode='r') as ari_file:
                ari = json.load(ari_file)
                all_aris.append(ari)
        return all_aris

    def calculate_mean_ARI_for_dataset(self, dataset_name):
        all_aris = self.read_all_ARIs(dataset_name)
        return self.calculate_mean_of_ari_lists(all_aris)

    def calculate_mean_of_ari_lists(self, all_aris):
        budget = self.query_budget
        ari_arr = np.zeros((len(all_aris), budget))
        i = 0
        for ari in all_aris:
            if len(ari) == 0:
                print("empty ari array! ignoring")
                continue
            if len(ari) > budget:
                print("more clusterings than querybudget, correcting by removing excess clusterings")
            elif len(ari) < budget:
                print("less clusterings than querybudget, correcting by repeating last ARI")

            ari = np.array(ari)
            # if the result is too long cut it off
            ari_arr[i, :len(ari)] = ari[:min(len(ari), budget)]
            # if the result is too short repeat the last ari till correct length
            ari_arr[i, len(ari):] = np.array([ari[-1]] * (budget - len(ari)))
            i+=1

        mean_ari = np.mean(ari_arr[:i,:], axis=0).tolist()
        return mean_ari

    def run(self):
        result_dict = dict()
        for dataset_name in os.listdir(self.get_aris_dir()):
            if dataset_name == '.DS_Store':
                continue
            mean_ARI = self.calculate_mean_ARI_for_dataset(dataset_name)
            result_dict[dataset_name] = mean_ARI
        all_aris = []
        for aris in result_dict.values():
            all_aris.append(aris)
        overall_mean_ari = self.calculate_mean_of_ari_lists(all_aris)
        result_dict["overall_average"] = overall_mean_ari
        os.makedirs(os.path.dirname(self.get_average_ari_file()), exist_ok=True)
        with open(self.get_average_ari_file(), mode = 'w') as average_ari_file:
            json.dump(result_dict, average_ari_file)


def generate_average_ari_tasks_for_tests(test_names, query_budget):
    for test_name in test_names:
        yield AverageARITask(test_name, query_budget)


def calculate_average_aris(test_names, query_budget):
    print("Calculating average ARIs")
    for task in tqdm(list(generate_average_ari_tasks_for_tests(test_names, query_budget))):
        task.run()

