import json
import os
from run_locally.run_tests import run_tests_from_generator
from config import COP_RF_LIKE_ARI_DIR, COP_RF_LIKE_AUA_DIR
from evaluate_clusterings.evaluation.area_under_ari_curve_coprf_like import get_area_under_ari_curve
class CalculateAreaUnderARITask:
    def __init__(self, test_name):
        self.test_name = test_name

    def get_available_dataset_names(self):
        common_dataset_names = None
        discovered_inconsistency = False
        for run_name in os.listdir(os.path.join(COP_RF_LIKE_ARI_DIR, self.test_name)):
            dataset_names = [filename[:-4] for filename in os.listdir(os.path.join(COP_RF_LIKE_ARI_DIR, self.test_name, run_name))]
            if common_dataset_names is None:
                common_dataset_names = set(dataset_names)
            else:
                common_dataset_names = common_dataset_names.intersection(dataset_names)
                if len(common_dataset_names) != len(dataset_names):
                    discovered_inconsistency = True
        if discovered_inconsistency:
            print("found inconsistency in available datasets for ", self.test_name)
            print("continuing with common datasets")
        return common_dataset_names

    @staticmethod
    def parse_run_name(run_name):
        query, noise = run_name.split(",")
        query_ratio, noise_ratio = float(query[2:-1]), int(noise[2:-1])
        return query_ratio,noise_ratio

    @staticmethod
    def get_run_name_from_dict(d):
        return CalculateAreaUnderARITask.get_run_name(d['query'], d['noise'])
    @staticmethod
    def get_run_name(query_ratio, noise_ratio):
        return "q={}%,n={}%".format(query_ratio, noise_ratio)

    def get_output_path(self):
        return os.path.join(COP_RF_LIKE_AUA_DIR, self.test_name+".txt")

    def get_data_format(self):
        noise_ratios = set()
        query_ratios = set()
        for run_name in os.listdir(os.path.join(COP_RF_LIKE_ARI_DIR, self.test_name)):
            query_ratio, noise_ratio = CalculateAreaUnderARITask.parse_run_name(run_name)
            noise_ratios.add(noise_ratio)
            query_ratios.add(query_ratio)
        if len(noise_ratios) == 1:
            fixed = "noise"
            changing = "query"
            fixed_value = list(noise_ratios)[0]
            changing_values = list(sorted(query_ratios))
        elif len(query_ratios) == 1:
            fixed = "query"
            changing = "noise"
            fixed_value = list(query_ratios)[0]
            changing_values = list(sorted(noise_ratios))
        else:
            raise Exception("both query and noise seem to have different values!")
        return fixed, fixed_value, changing, changing_values

    def get_data_array(self, dataset_name, fixed, fixed_value, changing, changing_values):
        aris = []
        for changing_value in changing_values:
            run_name = self.get_run_name_from_dict({fixed:fixed_value, changing:changing_value})
            filename = os.path.join(COP_RF_LIKE_ARI_DIR, self.test_name, run_name, dataset_name+".txt")
            os.makedirs(os.path.dirname(filename),exist_ok=True)
            with open(filename, mode = 'r') as ari_file:
                ari_number = json.load(ari_file)
            aris.append(ari_number)
        return aris


    def run(self):
        result_dict = {}
        fixed, fixed_value, changing, changing_values = self.get_data_format()
        for dataset_name in self.get_available_dataset_names():
            print(dataset_name)
            aris_for_dataset = self.get_data_array(dataset_name, fixed, fixed_value, changing, changing_values)
            aua = get_area_under_ari_curve(aris_for_dataset)
            result_dict[dataset_name] = aua
        with open(self.get_output_path(), mode = 'w') as result_file:
            json.dump((fixed,fixed_value, changing, changing_values, result_dict),result_file)


def generate_tasks(test_names):
    for test_name in test_names:
        task = CalculateAreaUnderARITask(test_name)
        if not os.path.isfile(task.get_output_path()):
            yield task

def calculate_area_under_ari_for(test_names, nb_of_cores = 3):
    test_generator = generate_tasks(test_names)
    print("calculating areas under ari")
    run_tests_from_generator(test_generator, nb_of_cores)

