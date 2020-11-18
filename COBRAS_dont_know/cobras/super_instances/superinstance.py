import abc


class SuperInstanceBuilder:
    @abc.abstractmethod
    def makeSuperInstance(self, data, indices, train_indices, parent = None):
        pass


class SuperInstance:
    """
    Class to represent a SuperInstance i.e. group of instances assumed to be in the same cluster.
        Attributes:
            data
            indices
            train_indices
            tried_splitting
            representative_idx
            children
            parent
        Functions:
            get_representative_idx()
            distance_to(SuperInstance)
            get_leaves()
            get_root()
            get_depth()
            copy()
    """
    def __init__(self, data, indices, train_indices, parent=None):
        if not isinstance(indices, list):
            raise ValueError('Should give a list of indices as input to SuperInstance...')

        self.data = data
        self.indices = indices
        self.train_indices = [x for x in indices if x in train_indices]
        self.tried_splitting = False

        self.representative_idx = None

        self.children = None
        self.parent = parent

    def get_representative_idx(self):
        try:
            return self.representative_idx
        except:
            raise ValueError('Super instances without training instances')

    @abc.abstractmethod
    def distance_to(self, other_superinstance):
        return

    def get_leaves(self):

        if self.children is None:
            return [self]
        else:
            d = []
            for s in self.children:
                d.extend(s.get_leaves())
            return d

    def get_root(self):
        if self.parent is None:
            return self
        else:
            return self.parent.get_root()

    def get_depth(self):
        if self.parent is None:
            return 1
        else:
            return self.parent.get_depth() + 1

    def copy(self):
        pass
