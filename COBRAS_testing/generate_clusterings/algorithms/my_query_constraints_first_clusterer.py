import time


class QueryFirstActiveClusterer:
    def __init__(self, clusterer):
        self.clusterer = clusterer

    def fit(self, X, nb_clusters, train_indices, querier):
        start_time = time.time()
        ml, cl = querier.get_constraints()
        clustering = self.clusterer.fit(X, ml, cl, nb_clusters)
        end_time = time.time()
        return [clustering.tolist()]*querier.nb_of_constraints , [end_time-start_time]*querier.nb_of_constraints, ml, cl
