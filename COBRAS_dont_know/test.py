from cobras.cobras import COBRAS
from cobras.cobras_logger import COBRASLogger
from cobras.constraints.constraint_type import ConstraintType
from heuristics.splitlevel_estimation_strategy import StandardSplitLevelEstimationStrategyAlwayskmeans
from heuristics.splitlevel_estimation_strategy import StandardSplitLevelEstimationStrategy
from cobras.querier.labelquerier import LabelQuerier
from util.datasets import Dataset
from heuristics.select_super_instance_heuristics import SelectMostInstancesHeuristic
from clustering_algorithms.kmeans_fixed_representative import KmeansFixedRepresentative
from cobras.super_instances.superinstance_select_representative import SuperInstance_select_representative_Builder
from util.visualizer import plot_clustering

def test_1(dataset_names):
    # dataset_names = ['faces_eyes_imagenet'] # Dataset.get_standard_dataset_names()
    for dataset_name in dataset_names:
        data = Dataset(dataset_name)
        querier = LabelQuerier(data.target, 100)
        clusterer = COBRAS(cobras_plus=True)
        logger = COBRASLogger()

        clusterings, runtimes, ml, cl, dk = clusterer.fit(data.data, None, None, querier, logger)

        counter = 0
        for clustering in clusterings:
            name = "clustering_"+str(counter)
            plot_clustering(data, clustering, name, True)
            counter += 1

        print("Queries done: ", len(logger.queried_constraints))
        print("predicted constraints: ", len(logger.predicted_constraints))
        print("MLs: ", len(ml))
        print("CLs: ", len(cl))

def test_2(datatset_names):
    for dataset_name in datatset_names:
        data = Dataset(dataset_name)
        querier = LabelQuerier(data.target, 100)
        clusterer = COBRAS(cobras_plus=True)
        logger = COBRASLogger()
        clusterings, runtimes, ml, cl, dk = clusterer.fit(data.data, None, None, querier, logger)

        final_clustering = clusterings[-1]
        total_predictions = len(logger.predicted_constraints)

        correct_predictions, incorrect_predictions = calculate_correct_predictions(logger.predicted_constraints,
                                                                                   data.target)

        print("Total predictions: {}\nCorrect predictions: {}\nFalse predictions: {}".format(total_predictions,
                                                                                             correct_predictions,
                                                                                             incorrect_predictions))
        print(len(ml)+len(cl))
def calculate_correct_predictions(predictions, target):
    correct = 0
    incorrect = 0
    for prediction in predictions:
        if target[prediction.i1] == target[prediction.i2] and prediction.type == ConstraintType.ML:
            correct += 1
        elif target[prediction.i1] != target[prediction.i2] and prediction.type == ConstraintType.CL:
            correct += 1
        else:
            incorrect += 1
    return correct, incorrect


def run_cobras():
    dataset = Dataset("ionosphere")
    querier = LabelQuerier(dataset.target, 100)
    splitstrat = StandardSplitLevelEstimationStrategyAlwayskmeans(SelectMostInstancesHeuristic())
    clusterer = COBRAS(cluster_algo=KmeansFixedRepresentative(),
                       superinstance_builder=SuperInstance_select_representative_Builder(),
                       splitlevel_strategy=splitstrat)
    print(clusterer.fit(dataset.data, None, None, querier))

if __name__ == '__main__':
    run_cobras()
