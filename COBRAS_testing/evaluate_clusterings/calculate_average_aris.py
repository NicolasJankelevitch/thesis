import numpy as np
from tqdm import tqdm

from config import FOLD_RESULT_DIR, COP_RF_LIKE_DIR
import os
import json
import statistics
from run_locally.run_tests import run_tests_from_generator

class COPRFLikeARIProcessing:
    def __init__(self, test_name, dataset_name):
        self.test_name = test_name
        self.dataset_name = dataset_name

    def average_aris_of_configuration(self, configuration):
        path = os.path.join(COP_RF_LIKE_DIR, self.test_name, "aris", self.dataset_name, configuration)
        aris = []
        for file in os.listdir(path):
            filepath = os.path.join(path, file)
            with open(filepath, mode = 'r') as ari_file:
                ari = json.load(ari_file)
            aris.append(ari)
        return statistics.mean(aris)

    def run(self):
        result_dict = dict()
        for configuration in os.listdir(os.path.join(COP_RF_LIKE_DIR, self.test_name,"aris", self.dataset_name)):
            mean_ari = self.average_aris_of_configuration(configuration)
            # parse configuration name in format "c{},n{}"
            constraints, noise = map(lambda x: float(x[1:]),configuration.split(","))
            result_dict[noise] = mean_ari
        ari_list_in_order = [v for k, v in sorted(result_dict.items())]
        result_file = os.path.join(COP_RF_LIKE_DIR, self.test_name, "average_aris", self.dataset_name + "_aris.txt")
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        with open(result_file, mode="w") as result:
            json.dump(ari_list_in_order, result)

class AverageARITask:
    def __init__(self,test_name, query_budget):
        self.test_name = test_name
        self.query_budget = query_budget

    def get_aris_dir(self):
        return os.path.join(FOLD_RESULT_DIR, self.test_name, "aris")

    def get_average_ari_file(self):
        return os.path.join(FOLD_RESULT_DIR, self.test_name, "ari_averages.txt")

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
        if self.query_budget is None:
            budget = min(len(ari) for ari in all_aris)
        else:
            budget = self.query_budget
        ari_arr = np.zeros((len(all_aris), budget))
        i = 0
        for ari in all_aris:
            if len(ari) == 0:
                print("empty ari array! ignoring")
                continue
            if len(ari) > budget:
                print("more clusterings ("+str(len(ari))+") than querybudget ("+str(budget)+"), correcting by removing excess clusterings "+self.test_name)
            elif len(ari) < budget:
                print("less clusterings ("+str(len(ari))+") than querybudget ("+str(budget)+"), correcting by repeating last ARI" + self.test_name)

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


def generate_average_ari_tasks_cop_rf(test_names):
    for test_name in test_names:
        path = os.path.join(COP_RF_LIKE_DIR, test_name, "aris")
        for dataset_name in os.listdir(path):
            yield COPRFLikeARIProcessing(test_name, dataset_name)

def calculate_average_aris_cop_rf(test_names, nb_of_cores = 8):
    tasks = list(generate_average_ari_tasks_cop_rf(test_names))
    run_tests_from_generator(tasks, nb_of_cores=nb_of_cores)

def generate_average_ari_tasks_for_tests(test_names, query_budget):
    for test_name in test_names:
        yield AverageARITask(test_name, query_budget)

def calculate_average_aris(test_names, query_budget):
    print("Calculating average ARIs")
    for task in tqdm(list(generate_average_ari_tasks_for_tests(test_names, query_budget))):
        task.run()

