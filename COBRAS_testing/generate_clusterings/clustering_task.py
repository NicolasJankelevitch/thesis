import copy
import functools
from pathlib import Path

import jsonpickle

from before_clustering.generate_folds import get_train_indices_for_dataset
from config import FOLD_RESULT_DIR


def cosc_result_extractor(cosc_clusterer):
    result_dict = dict()
    result_dict['no_cosc_result'] = cosc_clusterer.got_no_cosc_result
    return result_dict


def cobras_result_extractor(cobras_clusterer):
    result_dict = dict()
    logger = cobras_clusterer._cobras_log
    result_dict["user_constraints"] = logger.all_user_constraints
    result_dict["detected_noisy_data"] = logger.detected_noisy_constraint_data
    result_dict["phase_data"] = logger.algorithm_phases

    # to investigate noise correcting behaviour
    result_dict["corrected_constraint_sets"] = logger.corrected_constraint_sets
    return result_dict

def get_result_path(test_name, dataset_name, run_idx, fold_idx):
    return Path(FOLD_RESULT_DIR) / test_name / "clusterings" / dataset_name / f"run{run_idx}_fold{fold_idx}.txt"


def make_n_run_10_fold_cross_validation(test_name, clusterer, querier, dataset_names, n_runs, result_extractor = None):
    all_tasks = []
    for dataset_name in dataset_names:
        for run_idx in range(n_runs):
            for fold_idx in range(10):

                result_path = str(get_result_path(test_name, dataset_name, run_idx, fold_idx))
                if not Path(result_path).exists():
                    train_indices = get_train_indices_for_dataset(dataset_name, run_idx, fold_idx)
                    task = ClusteringTask(copy.deepcopy(clusterer), dataset_name, train_indices, result_extractor, copy.deepcopy(querier), result_path)
                    all_tasks.append(task)
    return all_tasks

make_n_run_10_fold_cross_validation_cobras = functools.partial(make_n_run_10_fold_cross_validation, result_extractor=cobras_result_extractor)

class ClusteringTask:
    def __init__(self, clusterer, dataset_name, training_indices, extra_result_extractor, querier, result_path):
        """
            The querier that is supplied does not need to have labels!
            These are filled in when the actual run method is called! (I know ugly but yeah...)
        """
        self.clusterer = clusterer
        self.dataset_name = dataset_name
        self.training_indices = training_indices
        self.result_extractor = extra_result_extractor
        self.querier = querier
        self.result_path = result_path


    def set_dataset(self, dataset):
        self.dataset = dataset

    def run(self, dataset):
        self.querier.set_labels_and_data(dataset.target, dataset.data)

        if hasattr(self.clusterer, "clustering_logger"):
            self.querier.logger = self.clusterer.clustering_logger
        if hasattr(self.clusterer, "raw_run_name"):
            self.clusterer.raw_run_name = str(self.result_path)
        all_clusters, runtimes, mls, cls, dks = self.clusterer.fit(dataset.data, dataset.number_of_classes(),
                                                              self.training_indices, self.querier)
        result_dict = dict()
        result_dict["clusterings"] = all_clusters
        result_dict["runtimes"] = runtimes
        result_dict["mls"] = mls
        result_dict["cls"] = cls
        result_dict["dks"] = dks
        result_dict["train_indices"] = self.training_indices
        result_dict["predicted"] = self.clusterer.logger.predicted_constraints
        result_dict["extra_asked"] = self.clusterer.logger.extra_asked

        # self.log_predicted_constraint((pair[0].i1, pair[0].i2, ctype))
        #self.extra_asked.append((pair[1].i1, pair[1].i2, ctype))
        # result_dict["reused_constraints"] = self.clusterer.logger.reused_constraints
        # result_dict["max_split_reached"] = self.clusterer.logger.max_split_reached
        # result_dict["split_levels"] = self.clusterer.logger.split_levels
        if self.result_extractor is not None:
            extra_result_dict = self.result_extractor(self.clusterer)
            result_dict.update(extra_result_dict)
        result_path = Path(self.result_path)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.result_path, mode='w') as output_file:
            string = jsonpickle.encode(result_dict)
            output_file.write(string)
