from active_semi_clustering.active.pairwise_constraints.example_oracle import MaximumQueriesExceeded


class DummyClusterer:
    def __init__(self):
        pass

    def fit(self, X, nb_clusters, train_indices, querier):
        clusterings = []
        while True:
            try:
                querier.query(1,2)
                clusterings.append([0]*len(X))
            except MaximumQueriesExceeded:
                break

        return clusterings, 1, 1,1