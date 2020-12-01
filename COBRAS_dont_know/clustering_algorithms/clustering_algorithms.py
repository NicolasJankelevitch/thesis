import abc

import numpy as np
from sklearn.cluster import KMeans


class ClusterAlgorithm:
    @abc.abstractmethod
    def cluster(self, data, indices, k, ml, cl, seed=None):
        pass

    def get_name(self):
        return type(self).__name__

    @classmethod
    def get_translated_constraints(cls, ml, cl, indices):
        filtered_ml = cls.__filter_constraint_set(ml, indices)
        filtered_cl = cls.__filter_constraint_set(cl, indices)
        translated_ml = cls.__translate_constraint_set(filtered_ml, indices)
        translated_cl = cls.__translate_constraint_set(filtered_cl, indices)
        return translated_ml, translated_cl

    @staticmethod
    def __filter_constraint_set(constraint_set, indices):
        filtered_set = set()
        for i1, i2 in constraint_set:
            if i1 in indices and i2 in indices:
                filtered_set.add((i1, i2))
        return filtered_set

    @staticmethod
    def __translate_constraint_set(constraint_set, indices):
        return set((indices.index(ml1), indices.index(ml2)) for ml1, ml2 in constraint_set)


class KMeansClusterAlgorithm(ClusterAlgorithm):
    def __init__(self, n_runs=10):
        self.n_runs = n_runs

    def cluster(self, data, indices, k, ml, cl, seed=None):
        if seed is not None:
            km = KMeans(k, n_init=self.n_runs, random_state=seed)
        else:
            km = KMeans(k, n_init=self.n_runs)

        # only cluster the given indices
        km.fit(data[indices, :])

        # return the labels as a list of integers
        return km.labels_.astype(np.int)
