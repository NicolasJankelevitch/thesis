from cobras.cobras_logger import COBRASLogger
from cobras.cobras import COBRAS
from cobras.querier.weak_querier import WeakQuerier
from util.datasets import Dataset
from sklearn import metrics


def run_cobras_on_dataset(dataset_name, number_of_queries):
    data = Dataset(dataset_name)

    clusterer = COBRAS(similarity_pred=True)
    querier = WeakQuerier(data.data, data.target, number_of_queries, 'local_nondet', max_prob=1)
    clusterings, runtimes, mls, cls, dks = clusterer.fit(data.data, None, None, querier)
    logger: COBRASLogger = clusterer.logger

    print("COBRAS finished in {}s with result {}".format(runtimes[-1], clusterings[-1]))
    print("Number of DK queries: {}".format(querier.total_DK))
    print("# of predicted constraints: {}".format(len(logger.predicted_constraints)))
    if len(logger.predicted_constraints) != 0:
        print("% of correct predictions: {}%".format(logger.n_correct_preds * 100 / len(logger.predicted_constraints)))
    print("ARI: ", metrics.adjusted_rand_score(data.target, clusterings[-1]))

    n_merging = logger.algorithm_phases.count("merging")
    n_splitting = logger.algorithm_phases.count("splitting")
    print("Merging ", n_merging)
    print("Splitting ", n_splitting)

if __name__ == '__main__':
    run_cobras_on_dataset("aggregation", 100)

