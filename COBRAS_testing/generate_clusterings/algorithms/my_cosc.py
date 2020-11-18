import copy
import datetime
import itertools
import math
import random
import os
import time
import traceback

import numpy as np
from scipy import sparse, spatial, io
from datetime import datetime
from config import TEMP_DIR, COSC_PATH
from datasets import Dataset

class NoCOSCResult(Exception):
    pass

class MyCOSCMatlab:
    def __init__(self, run_fast_version = False):
        self.W_cache = None
        self.run_fast_version = run_fast_version

    def signal_start(self, data):
        import matlab.engine
        import matlab
        start = time.time()
        self.engine = matlab.engine.start_matlab()
        print("engine startup took: ", time.time() - start)
        start = time.time()
        W = MyCOSCMatlab.construct_knn_matrix(data)
        self.engine.workspace["W"] = matlab.double(W.toarray().tolist())
        self.engine.eval('W=sparse(W);', nargout=0)
        print("calculating W took:", time.time() - start)
        # self.W_cache = W
        self.engine.addpath(COSC_PATH)

    def signal_end(self):
        self.engine.quit()

    def calculate_cacheable_info(self, X):
        pass

    def fit(self, X, ml, cl, nb_clusters):
        import matlab.engine
        import matlab
        # os.system("rm -rf " + os.path.dirname(
        #     os.path.abspath(__file__)) + '/tmp/' + hn + '/' + fn + '/clustering_result' + '_' + suffix + '.mat')
        # os.system("rm -rf " + os.path.dirname(
        #     os.path.abspath(__file__)) + '/tmp/' + hn + '/' + fn + '/matlab_input' + '_' + suffix + '.mat')

        # handle weight matrix calculation
        # for i1, i2 in itertools.chain(ml,cl):
        #     if not 0 <= i1 < X.shape[0] or not 0 <= i2 < X.shape[0]:
        #         print("X.shape = ", X.shape)
        #         print("W_size = ", self.engine.eval("size(W)", nargout=2))
        #         print("constraint: ", (i1,i2))
        #         print()
        self.engine.workspace["ML"] = matlab.double([[i1+1, i2+1] for i1,i2 in ml])
        self.engine.workspace["CL"] = matlab.double([[i1+1, i2+1] for i1,i2 in cl])
        self.engine.workspace["k"] = nb_clusters
        # result = False
        # for _ in range(3):
        try:
            if not self.run_fast_version:
                self.engine.run_and_save(nargout=0, stdout = None)
            else:
                self.engine.run_and_save_fast(nargout=0, stdout = None)
            # break
        except Exception as e:
            now = datetime.now()
            time_str=now.strftime("%d_%m_%Y_%H:%M:%S")
            randint = random.randint(0,10000000000)
            # self.engine.eval("save('{}/error_dump_{}_{}.mat')".format(TEMP_DIR,time_str, randint), nargout=0)
            with open("{}/exception_{}_{}.txt".format(TEMP_DIR, time_str, randint), mode="w") as file:
                traceback.print_exc(file = file )
            raise NoCOSCResult

            # if not result:

        clusters = self.engine.workspace["clusters"]
        # if len(clusters) == 1:
        #     print("LEN CLUSTERS IS 1")
        #     print(clusters)
        #     print(ml)
        #     print(cl)
        #     print(nb_clusters)
        # print("clusters:", len(clusters))
        # parse the result

        if len(clusters) == 1:
            pred = [int(c) for c in clusters[0]]
        else:
            pred = [int(clusters[i][0]) for i in range(len(X))]

        return pred 


    @staticmethod
    def construct_knn_matrix(X):
        """
        Constructing the symmetric KNN graph MATRIX as in Buhler et al. (Spectral Clustering based on the graph p-Laplacian)
        We choose K = 10 as in the paper.

        :param X: dataset for which knn matrix should be constructed
        :param ml: must-link constraints
        :param cl: cannot-link constraints
        :return: K-NN matrix
        """

        K = 10

        num_el = len(X)
        A = sparse.lil_matrix((num_el, num_el))

        start = time.time()
        tree = spatial.cKDTree(X)
        sigmas = np.zeros((num_el, 1))
        k_nbs = []

        start = time.time()
        for i in range(num_el):
            nbs = tree.query(X[i, :], K + 1)
            sigmas[i] = nbs[0][-1]
            k_nbs.append(nbs[1][1:])

        start = time.time()
        for i in range(num_el):
            for j in range(num_el):
                # if i < j:
                if i < j and ((i in k_nbs[j]) or (j in k_nbs[i])):
                    diff = np.array(X[i, :]) - np.array(X[j, :])
                    s_i = math.exp((-4.0 / (sigmas[i] ** 2)) * np.dot(diff, diff))
                    s_j = math.exp((-4.0 / (sigmas[j] ** 2)) * np.dot(diff, diff))
                    A[i, j] = max(s_i, s_j)
                    if np.isnan(A[i, j]):
                        A[i, j] = 1.0

                    A[j, i] = A[i, j]

        return A.tocsr()

if __name__=='__main__':
    dataset = Dataset("iris")
    clusterer = MyCOSCMatlab()
    result = clusterer.fit(dataset.data, [(1,2),(1,5)],[(3,4),(5,10)], dataset.number_of_classes())
    print("got result", result)