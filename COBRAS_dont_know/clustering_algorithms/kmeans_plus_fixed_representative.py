from clustering_algorithms.clustering_algorithms import ClusterAlgorithm
import numpy as np
from copy import deepcopy


class KmeansPlusFixedRepresentative(ClusterAlgorithm):
    def __init__(self, n_runs=10):
        self.n_runs = n_runs
        self.parent_repr_idx = None

    # This code was adapted from https://www.kaggle.com/andyxie/k-means-clustering-implementation-in-python
    def cluster(self, data, indices, k, ml, cl, seed=None):
        k = k+1
        if self.parent_repr_idx is None:
            raise Exception("parent_representative_idx is not set (this parameter is required for this clutering technique)")

        used_data = data[indices, :]
        fixed_center = data[self.parent_repr_idx]

        # nr of training data
        n = used_data.shape[0]
        # nr of features
        c = used_data.shape[1]

        # generate random centers, here we use sigma and mean to ensure it represents the whole data
        mean = np.mean(used_data, axis=0)
        std = np.std(used_data, axis=0)
        centers = np.random.randn(k - 1, c) * std + mean
        centers = np.vstack([centers, fixed_center])

        centers_old = np.zeros(centers.shape)  # used to store old centers
        centers_new = deepcopy(centers)  # Store new centers

        clusters = np.zeros(n)
        distances = np.zeros((n, k))

        error = np.linalg.norm(centers_new - centers_old)
        n_runs = 10
        # When, after an update, the estiame of that center stays the same, exit loop
        runs = 0
        while error != 0 and runs != n_runs:
            # Measure the distance to every center
            for i in range(k):
                distances[:, i] = np.linalg.norm(used_data - centers[i], axis=1)

            clusters = np.argmin(distances, axis=1)
            centers_old = deepcopy(centers_new)

            for i in range(k - 1):
                centers_new[i] = np.mean(used_data[clusters == i], axis=0)
            centers_new[k - 1] = centers_old[k - 1]
            error = np.linalg.norm(centers_new - centers_old)
            runs = runs + 1

        return clusters
