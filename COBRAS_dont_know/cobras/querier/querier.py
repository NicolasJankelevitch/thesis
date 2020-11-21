import abc


class Querier:
    """
        Querier base class.
    """
    def __init__(self):
        pass

    @abc.abstractmethod
    def query(self, idx1, idx2):
        return

    @abc.abstractmethod
    def query_limit_reached(self):
        return

    @abc.abstractmethod
    def set_labels_and_data(self, labels=None, data=None):
        return

