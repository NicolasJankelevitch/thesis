import os

from scipy import io

from config import COSC_MATRICES_DIR
from datasets import Dataset
from generate_clusterings.algorithms.my_cosc import MyCOSCMatlab
from run_locally.run_tests import run_tests_from_generator


class CalculateWMatrix:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def run(self):
        data = Dataset(self.dataset_name)
        weight_matrix = MyCOSCMatlab.construct_knn_matrix(data.data)
        output_file_name = os.path.join(COSC_MATRICES_DIR, data.name + "_W.mat")
        mdict = dict()
        mdict["W"] = weight_matrix
        os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
        io.savemat(output_file_name, mdict)


def calculate_all_W_matrices(nb_of_cores=8):
    tasks = []
    for dataset_name in Dataset.get_dataset_names():
        tasks.append(CalculateWMatrix(dataset_name))
    run_tests_from_generator(tasks, nb_of_cores=nb_of_cores)

if __name__=='__main__':
    calculate_all_W_matrices(8)