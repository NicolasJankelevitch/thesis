from generate_clusterings.algorithms.mpck.mpckmeans import MPCKMeans
from active_semi_clustering import MPCKMeansMF

class MyMPCKMeans():
    def __init__(self, w = 1, max_iter = 10, learn_multiple_full_matrices = True):
        """

        :param w:
        :param max_iter:
        :param learn_multiple_full_matrices:
            False: learn a single diagonal matrix as distance metric --> this corresponds with feature weighting
            True: learn a full matrix per cluster a full matrix can make new features as a linear combination of existing features
        """
        self.w = w
        self.max_iter = max_iter
        self.learn_multiple_matrices = learn_multiple_full_matrices

    def fit(self, X, ml, cl, nb_clusters):
        if self.learn_multiple_matrices:
            clusterer = MPCKMeansMF(n_clusters = nb_clusters, w = self.w , max_iter = self.max_iter)
        else:
            clusterer = MPCKMeans(n_clusters=nb_clusters, w = self.w, max_iter=self.max_iter)
        clusterer.fit(X, None, ml, cl)
        return clusterer.labels_


