from cobras.cobras import COBRAS
from cobras.cobras_logger import COBRASLogger
from heuristics.splitlevel_estimation_strategy import StandardSplitLevelEstimationStrategyAlwayskmeans
from cobras.querier.labelquerier import LabelQuerier
from util.datasets import Dataset
from heuristics.select_super_instance_heuristics import SelectMostInstancesHeuristic
from clustering_algorithms.kmeans_fixed_representative import KmeansFixedRepresentative
from cobras.super_instances.superinstance_select_representative import SuperInstance_select_representative_Builder


def test_1():
    dataset_names = Dataset.get_standard_dataset_names()
    for dataset_name in dataset_names:
        data = Dataset(dataset_name)
        querier = LabelQuerier(data.target, 100)
        splitstrat = StandardSplitLevelEstimationStrategyAlwayskmeans(SelectMostInstancesHeuristic())
        clusterer = COBRAS(cluster_algo=KmeansFixedRepresentative(),
                           superinstance_builder=SuperInstance_select_representative_Builder(),
                           splitlevel_strategy=splitstrat)
        logger = COBRASLogger()

        clusterings, runtimes, ml, cl, dk = clusterer.fit(data.data, None, None, querier, logger)

        val = len(ml) + len(cl) + len(dk)
        print("{}:\t{}".format(dataset_name, val))


if __name__ == '__main__':
    test_1()
