import itertools
from math import log
import numpy as np
import numpy.random as rnd
class ClusteringWithNoisyQueries:
    def __init__(self, noise_probability):
        self.noise_p = noise_probability
        pass

    def fit(self, X, nb_clusters, train_indices, querier):
        c = 16/(1-self.noise_p)^2
        nb_instances = X.shape[0]
        c_log_n =  c * log(nb_instances)
        nb_of_random_instances = c_log_n
        V_prime = rnd.random_integers(0,nb_instances-1,nb_of_random_instances)
        for i1, i2 in itertools.combinations(V_prime,2):
            constraint = querier._query_points(i1, i2)

