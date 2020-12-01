import numpy as np

from cobras.super_instances.superinstance import SuperInstance, SuperInstanceBuilder


class SuperInstance_select_representative_Builder(SuperInstanceBuilder):
    def __init__(self):
        pass

    def makeSuperInstance(self, data, indices, train_indices, parent=None):
        return SuperInstance_select_representative(data, indices, train_indices, parent)


class SuperInstance_select_representative(SuperInstance):
    def __init__(self, data, indices, train_indices, parent=None):
        super(SuperInstance_select_representative, self).__init__(data, indices, train_indices, parent)

        self.centroid = np.mean(data[indices, :], axis=0)

        self.si_train_indices = [x for x in indices if x in train_indices]

        if len(set(self.si_train_indices)) < len(self.si_train_indices):
            print("something goes wrong!")
        try:
            # Representative instance is the training instance that is closest to the clusters centroid
            if parent is not None:
                if parent.representative_idx in indices:
                    self.representative_idx = parent.representative_idx
                else:
                    self.representative_idx = min(self.si_train_indices,
                                                  key=lambda x: np.linalg.norm(self.data[x, :] - self.centroid))
            else:
                self.representative_idx = min(self.si_train_indices,
                                              key=lambda x: np.linalg.norm(self.data[x, :] - self.centroid))

        except:
            raise ValueError('Super instance without training instances')

    def distance_to(self, other_superinstance):
        return np.linalg.norm(self.centroid - other_superinstance.centroid)

    def copy(self):
        return SuperInstance_select_representative(self.data, self.indices, self.train_indices)
