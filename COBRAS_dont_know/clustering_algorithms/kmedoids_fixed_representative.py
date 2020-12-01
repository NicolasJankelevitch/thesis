from clustering_algorithms.clustering_algorithms import ClusterAlgorithm
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


class KmedoidsFixedRepresentative(ClusterAlgorithm):
    def __init__(self, n_runs=10):
        self.n_runs = n_runs
        self.parent_repr_idx = None

    # This code was adapted from https://github.com/letiantian/kmedoids/blob/master/kmedoids.py
    def cluster(self, data, indices, k, ml, cl, seed=None):
        if self.parent_repr_idx is None:
            raise Exception("parent_representative_idx is not an optional parameter")

        r = data[self.parent_repr_idx]
        temp = data[indices, :]

        fixed = None
        if r.tolist() in temp.tolist():
            fixed = temp.tolist().index(r.tolist())
            k = k-1

        D = pairwise_distances(temp, metric='euclidean')
        # determine dimensions of distance matrix D
        m, n = D.shape

        if k > n:
            raise Exception('too many medoids')

        # find a set of valid initial cluster medoid indices since we
        # can't seed different clusters with two points at the same location
        valid_medoid_inds = set(range(n))
        if fixed is not None:
            invalid_medoid_inds = set([fixed])
        else:
            invalid_medoid_inds = set([])
        # rs = rows, cs=cols. using zip, this gives coordinates of elements that satisfy the given condition (D==0)
        rs, cs = np.where(D == 0)
        # the rows, cols must be shuffled because we will keep the first duplicate below
        index_shuf = list(range(len(rs)))
        np.random.shuffle(index_shuf)
        rs = rs[index_shuf]
        cs = cs[index_shuf]
        for r, c in zip(rs, cs):
            # if there are two points with a distance of 0...
            # keep the first one for cluster init
            if r < c and r not in invalid_medoid_inds:
                invalid_medoid_inds.add(c)
        valid_medoid_inds = list(valid_medoid_inds - invalid_medoid_inds)
        if k > len(valid_medoid_inds):
            raise Exception('too many medoids (after removing {} duplicate points)'.format(
                len(invalid_medoid_inds)))

        # randomly initialize an array of k medoid indices
        # add the fixed mediod
        M = np.array(valid_medoid_inds)
        np.random.shuffle(M)
        M = M[:k]
        if fixed is not None:
            M = np.append(M, fixed)
            k=k+1
        M = np.sort(M)
        # create a copy of the array of medoid indices
        Mnew = np.copy(M)
        # initialize a dictionary to represent clusters
        C = {}
        for t in range(self.n_runs):
            # determine clusters, i. e. arrays of data indices
            J = np.argmin(D[:, M], axis=1)
            for kappa in range(k):
                C[kappa] = np.where(J == kappa)[0]
            # update cluster medoids
            for kappa in range(k):
                J = np.mean(D[np.ix_(C[kappa], C[kappa])], axis=1)
                j = np.argmin(J)
                if (fixed is not None) and not np.equal(Mnew[kappa], np.int32(fixed)):
                    Mnew[kappa] = C[kappa][j]
            np.sort(Mnew)
            # check for convergence
            if np.array_equal(M, Mnew):
                break
            M = np.copy(Mnew)
        else:
            # final update of cluster memberships
            J = np.argmin(D[:, M], axis=1)
            for kappa in range(k):
                C[kappa] = np.where(J == kappa)[0]
                # return results

        array = np.zeros(len(indices), dtype=int)
        for key in C:
            for x in indices:
                if x in C[key]:
                    array[x] = key

        return array
