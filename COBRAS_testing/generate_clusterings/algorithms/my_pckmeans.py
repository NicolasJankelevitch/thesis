from active_semi_clustering import PCKMeans


class MyPCKMeans:
    def __init__(self, w = 1, max_iter = 100):
        self.w = w
        self.max_iter = max_iter

    def fit(self, X, ml, cl, nb_clusters):
        clusterer = PCKMeans(nb_clusters,self.max_iter, self.w)
        clusterer.fit(X,None, ml,cl)
        return clusterer.labels_
