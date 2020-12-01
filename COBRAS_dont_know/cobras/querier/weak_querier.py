from cobras.constraints.constraint_type import ConstraintType
from .querier import *
from .maximum_queries_exceeded import *
import numpy as np


class WeakQuerierBuilder:
    def __init__(self, oracle_type, q, rho, nu, max_prob, max_queries):
        self.oracle_type = oracle_type
        self.q = q
        self.rho = rho
        self.nu = nu
        self.max_queries = max_queries
        self.max_prob = max_prob

    def build_querier(self, dataset):
        return WeakQuerier(dataset.data, dataset.target, self.max_queries,
                           self.oracle_type, self.q, self.rho, self.nu,
                           self.max_prob)


class WeakQuerier(Querier):
    """
        Querier allowing dont-know answers, it has knowledge of the ground truth.
        Taken into assumption local distance-weak oracles for the uncertainty
        from this paper: https://arxiv.org/abs/1711.07433
        Attributes:
            data            Training data
            labels          True labels of training data
            max_queries     Max number of queries allowed for COBRAS
            queries_asked   Counter of already-asked queries
            oracle_type    Type of weak oracle [random, local] (Relaxed oracles paper)
    """

    # TODO: May be useful to refactor this class, for easier reading. Return number to type

    def __init__(self, data, labels, maximum_number_of_queries, oracle_type='random',
                 q=1, rho=0.8, nu=1.25, max_prob=0.95):
        super().__init__()
        self.labels = labels
        self.data = data
        self.max_queries = maximum_number_of_queries
        self.queries_asked = 0
        self.oracle_type = oracle_type
        # Distance-weak oracle parameters
        self.q = q
        self.rho = rho
        self.nu = nu
        # Non-deterministic
        self.max_prob = max_prob
        self.min_prob = 1 - max_prob
        # Ground truth centroids and radius for distance calculation.
        self.gt_centroids = {}
        self.gt_radius = {}
        # range(int(min(labels)), int(max(labels))+1)
        # Number of each type of queries answered
        self.total_ML = 0
        self.total_CL = 0
        self.total_DK = 0

    def query_limit_reached(self):
        return self.queries_asked >= self.max_queries

    def query(self, i: int, j: int) -> ConstraintType:
        """
        Query the relation of two instances
        :param i: Index of instance
        :param j: Index of instance
        :return:
            ConstraintType (ML, CL, DK)
        """
        # if self.__calculated is False:
        #     self.calculate_radius_centroids()

        if self.query_limit_reached():
            raise MaximumQueriesExceeded
        self.queries_asked += 1

        # Local-distance weak oracle, totally deterministic. Dependent on rho and nu.
        if self.oracle_type == "local":
            d_xy = np.linalg.norm(self.data[i, :] - self.data[j, :])
            if self.labels[i] == self.labels[j]:
                if d_xy > 2 * self.rho * self.gt_radius[self.labels[i]]:
                    constraint_type = 0
                else:
                    constraint_type = 1
            else:
                d_xi = np.linalg.norm(self.data[i, :] - self.gt_centroids[self.labels[i]])
                d_yi = np.linalg.norm(self.data[j, :] - self.gt_centroids[self.labels[j]])
                if (self.nu - 1) * min(d_xi, d_yi) > d_xy:
                    constraint_type = 0
                else:
                    constraint_type = -1

        # Local-distance weak oracle, non-deterministic. Based on exponential distribution.
        elif self.oracle_type == "local_nondet":
            b = 25
            d_xy = np.linalg.norm(self.data[i, :] - self.data[j, :])
            if self.labels[i] == self.labels[j]:
                prob_dk = (self.max_prob - self.min_prob) / (b ** (2 * self.gt_radius[self.labels[i]]) - 1)
                prob_dk *= (b ** d_xy - 1)
                prob_dk += self.min_prob
                if np.random.rand() <= prob_dk:
                    constraint_type = 0
                else:
                    constraint_type = 1
            else:
                d_centroids = np.linalg.norm(self.gt_centroids[self.labels[i]] - self.gt_centroids[self.labels[j]])
                prob_dk = (self.min_prob - self.max_prob) / (b ** d_centroids - 1)
                prob_dk *= (b ** d_xy - 1)
                prob_dk += self.max_prob
                if d_centroids < d_xy or np.random.rand() <= prob_dk:
                    constraint_type = -1
                else:
                    constraint_type = 0

        # Random uncertain constraints with probability 1-q
        elif self.oracle_type == "random":
            constraint_type = np.random.binomial(1, self.q) * 2 * (int(self.labels[i] == self.labels[j]) - 0.5)
        else:
            constraint_type = 2 * (int(self.labels[i] == self.labels[j]) - 0.5)

        if constraint_type == 1:
            self.total_ML += 1
            return ConstraintType.ML
        elif constraint_type == -1:
            self.total_CL += 1
            return ConstraintType.CL
        elif constraint_type == 0:
            self.total_DK += 1
            return ConstraintType.DK

    def set_labels_and_data(self, labels=None, data=None):
        if labels is not None:
            self.labels = labels
        if data is not None:
            self.data = data
        self.calculate_radius_centroids()

    def calculate_radius_centroids(self):
        if self.data is not None and self.labels is not None:
            for i in np.unique(self.labels):
                self.gt_centroids[i] = (np.mean(self.data[self.labels == i,], axis=0))
                max_rad = max(np.linalg.norm(self.data[self.labels == i, :] - self.gt_centroids[i], axis=1))
                self.gt_radius[i] = max_rad
