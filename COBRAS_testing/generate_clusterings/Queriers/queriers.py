import random

import numpy as np

from cobras_ts.querier.querier import MaximumQueriesExceeded


class ProbabilisticNoisyQuerier():
    def __init__(self, labels, noise_percentage, maximum_number_of_queries):
        self.labels = labels
        self.noise_percentage = noise_percentage
        self.max_queries = maximum_number_of_queries
        self.queries_asked = 0

    def update_clustering(self, clustering):
        pass

    def query_limit_reached(self):
        return self.queries_asked >= self.max_queries


    def query_points(self, i, j):
        return self.query(i,j)

    def query(self, i, j):
        if self.queries_asked >= self.max_queries:
            raise MaximumQueriesExceeded
        correct_answer = self.labels[i] == self.labels[j]
        if random.random()<self.noise_percentage:
            answer = not correct_answer
        else:
            answer = correct_answer
        self.queries_asked += 1
        return answer


class FixedNoisyQuerier:
    @staticmethod
    def generate_noisy_index_set(nb_noisy_constraints, query_limit, random_seed):
        all_choices = range(0,query_limit)
        np.random.seed(random_seed)
        return set(np.random.choice(all_choices,nb_noisy_constraints, replace = False))

    def __init__(self, labels, nb_noisy_constraints, query_limit, random_seed=None):
        self.labels = labels
        self.noisy_index_set = FixedNoisyQuerier.generate_noisy_index_set(nb_noisy_constraints,query_limit, random_seed)
        self.max_queries = query_limit
        self.queries_asked = 0

    def update_clustering(self, clustering):
        pass

    def query_limit_reached(self):
        return self.queries_asked >= self.max_queries

    def query_points(self, i, j):
        return self.query(i,j)

    def query(self, i, j):
        if self.queries_asked >= self.max_queries:
            raise MaximumQueriesExceeded
        correct_answer = self.labels[i] == self.labels[j]
        if self.queries_asked in self.noisy_index_set:
            answer = not correct_answer
        else:
            answer = correct_answer
        self.queries_asked += 1
        return answer












