from clustering_algorithms.clustering_algorithms import ClusterAlgorithm


class KmeansFixedRepresentative(ClusterAlgorithm):
    def __init__(self, n_runs=10):
        self.n_runs = n_runs

    def cluster(self, data, indices, k, ml, cl, seed=None):
        print("todo")