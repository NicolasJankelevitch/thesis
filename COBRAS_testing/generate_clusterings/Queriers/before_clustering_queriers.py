import itertools
import numpy as np

class RandomQuerierBuilder:
    def __init__(self, seed, nb_of_constraints, noise_percentage):
        self.seed = seed
        self.nb_of_constraints = nb_of_constraints
        self.noise_percentage = noise_percentage

    def build_querier(self, dataset):
        return RandomQuerier(dataset.target, self.seed, self.nb_of_constraints, self.noise_percentage)

class RandomQuerier:
    def __init__(self, labels, seed, nb_of_constraints, noise_percentage):
        self.labels = labels
        self.seed = seed
        self.nb_of_constraints = nb_of_constraints
        self.noise_percentage = noise_percentage

    def query_constraint(self, i,j):
        return self.labels[i] == self.labels[j]

    def get_constraints(self):
        all_pairs = list(itertools.combinations(range(len(self.labels)),2))
        random= np.random.RandomState(self.seed)
        constraints_to_query_indices = random.choice(range(len(all_pairs)), self.nb_of_constraints, replace=False)
        constraints_to_query = np.array(all_pairs)[constraints_to_query_indices]
        nb_of_noisy_constraints = int(self.noise_percentage * self.nb_of_constraints)
        noisy_indices = set(random.choice(range(len(self.labels)), nb_of_noisy_constraints, replace = False))
        ml, cl = [], []
        for idx, (i1,i2) in enumerate(constraints_to_query):
            correct_answer = self.query_constraint(i1,i2)
            if idx in noisy_indices:
                answer = not correct_answer
            else:
                answer = correct_answer
            if answer:
                ml.append((int(i1),int(i2)))
            else:
                cl.append((int(i1),int(i2)))
        return ml, cl
