from cobras.cobras import COBRAS
from cobras.cobras_logger import COBRASLogger
from heuristics.splitlevel_estimation_strategy import StandardSplitLevelEstimationStrategyAlwayskmeans
from heuristics.splitlevel_estimation_strategy import StandardSplitLevelEstimationStrategy
from cobras.querier.labelquerier import LabelQuerier
from util.datasets import Dataset
from heuristics.select_super_instance_heuristics import SelectMostInstancesHeuristic
from clustering_algorithms.kmeans_fixed_representative import KmeansFixedRepresentative
from cobras.super_instances.superinstance_select_representative import SuperInstance_select_representative_Builder


def test_1(dataset_names):
    # dataset_names = ['faces_eyes_imagenet'] # Dataset.get_standard_dataset_names()
    for dataset_name in dataset_names:
        data = Dataset(dataset_name)
        querier = LabelQuerier(data.target, 100)
        clusterer = COBRAS(cobras_plus=True)
        logger = COBRASLogger()

        clusterings, runtimes, ml, cl, dk = clusterer.fit(data.data, None, None, querier, logger)


if __name__ == '__main__':
    test_1(["iris"])
