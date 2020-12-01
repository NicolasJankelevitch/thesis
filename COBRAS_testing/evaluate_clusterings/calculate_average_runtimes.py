import json
import os

import jsonpickle
from tqdm import tqdm

from config import FOLD_RESULT_DIR

class AverageRuntimeTask:
    def __init__(self,test_name, query_budget):
        self.test_name = test_name
        self.query_budget = query_budget

    def get_aris_dir(self):
        return os.path.join(FOLD_RESULT_DIR, self.test_name, "clusterings")

    def get_average_runtime_file(self):
        return os.path.join(FOLD_RESULT_DIR, self.test_name, f"runtime_averages_{self.query_budget}.txt")

    def read_all_runtimes(self, dataset_name):
        dataset_path = os.path.join(self.get_aris_dir(), dataset_name)
        all_runtimes = []
        for filename in os.listdir(dataset_path):
            with open(os.path.join(dataset_path,filename), mode='r') as runtime_file:
                string = runtime_file.readline()
                result_dir = jsonpickle.decode(string)
                runtimes = result_dir["runtimes"]
                if len(runtimes) == 0:
                    print("empty runtime array! in "+dataset_path+filename)
                else:
                    if self.query_budget is None:
                        all_runtimes.append(runtimes[-1])
                    else:
                        all_runtimes.append(runtimes[min(self.query_budget, len(runtimes))-1])

                    all_runtimes.append(runtimes[-1])
        return all_runtimes

    def calculate_mean_runtime_for_dataset(self, dataset_name):
        all_runtimes = self.read_all_runtimes(dataset_name)
        return sum(all_runtimes)/len(all_runtimes)

    def run(self):
        result_dict = dict()
        for dataset_name in os.listdir(self.get_aris_dir()):
           mean_runtime = self.calculate_mean_runtime_for_dataset(dataset_name)
           result_dict[dataset_name] = mean_runtime
        os.makedirs(os.path.dirname(self.get_average_runtime_file()), exist_ok=True)
        with open(self.get_average_runtime_file(), mode ='w') as average_ari_file:
            json.dump(result_dict, average_ari_file)

class AverageRuntimeTaskOLD:
    def __init__(self,test_name, query_budget):
        self.test_name = test_name
        self.query_budget = query_budget

    def get_aris_dir(self):
        return os.path.join(FOLD_RESULT_DIR, self.test_name, "clusterings")

    def get_average_runtime_file(self):
        return os.path.join(FOLD_RESULT_DIR, self.test_name, f"runtime_averages_{self.query_budget}.txt")

    def read_all_runtimes(self, dataset_name):
        dataset_path = os.path.join(self.get_aris_dir(), dataset_name)
        all_runtimes = []
        for filename in os.listdir(dataset_path):
            with open(os.path.join(dataset_path,filename), mode='r') as runtime_file:
                execution_info = json.load(runtime_file)
                runtimes = execution_info[1]
                if len(runtimes) == 0:
                    print("empty runtime array! in "+dataset_path+filename)
                else:
                    if self.query_budget is None:
                        all_runtimes.append(runtimes[-1])
                    else:
                        all_runtimes.append(runtimes[min(self.query_budget, len(runtimes))-1])

                    all_runtimes.append(runtimes[-1])
        return all_runtimes

    def calculate_mean_runtime_for_dataset(self, dataset_name):
        all_runtimes = self.read_all_runtimes(dataset_name)
        return sum(all_runtimes)/len(all_runtimes)

    def run(self):
        result_dict = dict()
        for dataset_name in os.listdir(self.get_aris_dir()):
           mean_runtime = self.calculate_mean_runtime_for_dataset(dataset_name)
           result_dict[dataset_name] = mean_runtime
        os.makedirs(os.path.dirname(self.get_average_runtime_file()), exist_ok=True)
        with open(self.get_average_runtime_file(), mode ='w') as average_ari_file:
            json.dump(result_dict, average_ari_file)

def generate_average_runtime_tasks_for_tests(test_names, query_budget, recalculate, use_old):
    for test_name in test_names:
        if use_old:
            task = AverageRuntimeTaskOLD(test_name, query_budget)
        else:
            task = AverageRuntimeTask(test_name, query_budget)
        file_exists =os.path.exists(task.get_average_runtime_file())
        if not file_exists or (file_exists and recalculate):
            yield task

def calculate_average_runtimes_for_tests(test_names, query_budget, recalculate = False, use_old = False):
    print("Calculating average runtimes")
    for task in tqdm(generate_average_runtime_tasks_for_tests(test_names,query_budget, recalculate, use_old), total = len(test_names)):
        task.run()
