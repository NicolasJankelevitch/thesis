from generate_clusterings.Queriers.queriers import ProbabilisticNoisyQuerier, FixedNoisyQuerier


class ProbabilisticNoisyQuerierBuilder:
    def __init__(self, noise_percentage, max_queries):
        self.noise_percentage = noise_percentage
        self.max_queries = max_queries

    def build_querier(self, dataset):
        return ProbabilisticNoisyQuerier(dataset.target, noise_percentage=self.noise_percentage, maximum_number_of_queries=self.max_queries)

class FixedNoisyQuerierBuilder:
    def __init__(self, max_nb_queries, noisy_queries):
        self.max_nb_queries = max_nb_queries
        self.noisy_queries = noisy_queries

    def build_querier(self, dataset):
        return FixedNoisyQuerier(dataset.target,self.noisy_queries,self.max_nb_queries)