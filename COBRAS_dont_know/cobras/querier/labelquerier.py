from cobras.constraints.constraint_type import ConstraintType
from .maximum_queries_exceeded import *
from .querier import *


class LabelQuerier(Querier):
    """
        Querier that _always_ answers correctly the relation of two instances.
        Attributes:
            labels:         Ground truth labels of data.
            max_queries:    Maximun number of queries to answer by the querier.
            queries_asked:  Counter of already answered queries.
    """
    def __init__(self, labels, maximum_number_of_queries):
        super().__init__()
        self.labels = labels
        self.max_queries = maximum_number_of_queries
        self.queries_asked = 0

    def query_limit_reached(self):
        return self.queries_asked >= self.max_queries

    def query(self, i, j):
        """
                Query the relation of two instances
                :param i: Index of instance
                :param j: Index of instance
                :return:
                    ConstraintType (ML, CL)
        """
        if self.query_limit_reached():
            raise MaximumQueriesExceeded
        self.queries_asked += 1
        if self.labels[i] == self.labels[j]:
            return ConstraintType.ML
        else:
            return ConstraintType.CL

