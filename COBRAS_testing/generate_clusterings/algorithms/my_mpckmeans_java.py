import datetime
import os
import random
import socket
import subprocess

from config import TEMP_DIR,WEKA_PATH
from datasets import Dataset


class MyMPCKMeansJava:
    def __init__(self, max_iterations = 200, learn_multiple_matrices = False, w = 1, take_transitive_closure = True ):
        """

        :param w:
        :param max_iter:
        :param learn_multiple_full_matrices:
            False: learn a single diagonal matrix as distance metric --> this corresponds with feature weighting
            True: learn a full matrix per cluster a full matrix can make new features as a linear combination of existing features
        """
        self.max_iterations = max_iterations
        self.w = w
        self.max_iter = max_iterations
        self.learn_multiple_matrices = learn_multiple_matrices
        self.take_transitive_closure = take_transitive_closure

    @staticmethod
    def write_constraints(ml, cl, fn):
        f = open(fn, 'w+')

        for c in cl:
            f.write(str(c[0]) + '\t' + str(c[1]) + '\t-1\n')

        for c in ml:
            f.write(str(c[0]) + '\t' + str(c[1]) + '\t1\n')

        f.close()

    @staticmethod
    def write_arff(X, labels, fn):
        f = open(fn, 'w+')

        f.write('@RELATION myrel\n\n')

        for i in range(len(X[0])):
            f.write('@ATTRIBUTE attr_' + str(i) + ' REAL\n')

        f.write('@ATTRIBUTE class {')
        uniq_labels = list(set(labels))
        for cls in uniq_labels[:-1]:
            f.write(str(int(cls)) + ",")
        f.write(
            str(int(uniq_labels[-1])) + '}\n\n')  # class shouldn't matter (we will ignore MPCKMeans accuracy estimates)

        f.write('@DATA\n')

        for vec_idx in range(X.shape[0]):
            for el in X[vec_idx, :]:
                f.write(str(el) + ",")
            f.write(str(int(labels[vec_idx])) + '\n')  # class

        f.close()

    @staticmethod
    def parse_result(fn, X):
        pred = [-1] * len(X)
        # metric = np.zeros((len(X[0]), len(X[0])))

        f = open(fn, mode="r")
        # read_metric_diagonal = False
        # read_metric_full = False
        # diag_ctr = 0
        for line in f:
            els = line.split('\t')
            pred[int(els[0].strip())] = int(els[1].strip())

        f.close()
        return pred

    def fit(self, X, ml, cl, nb_clusters):
        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S%f") + str(random.randint(0, 10000))

        hn = socket.gethostname()
        os.makedirs(TEMP_DIR, exist_ok=True)
        input = os.path.join(TEMP_DIR, "mpckmeans_input_" + suffix + ".arff")
        constraints = os.path.join(TEMP_DIR, "constraints_" + suffix + ".constraints")
        output = os.path.join(TEMP_DIR, "mpckmeans_result_" + suffix+".txt")
        nb_of_instances = X.shape[0]

        MyMPCKMeansJava.write_arff(X, [0] * nb_of_instances, input)
        MyMPCKMeansJava.write_constraints(ml, cl, constraints)
        additional_arguments = ""
        if self.learn_multiple_matrices:
            additional_arguments+="-U "
        if not self.take_transitive_closure:
            additional_arguments+="-V "

        os.system(
            f"CLASSPATH={WEKA_PATH} java weka/clusterers/MPCKMeans -D {input} -C {constraints} -O {output} -N {nb_clusters} -i {self.max_iterations} -m {self.w} -c {self.w} {additional_arguments}> /dev/null 2>&1"
        )
        result = MyMPCKMeansJava.parse_result(output, X)

        # now remove the input and output files
        os.remove(input)
        os.remove(constraints)
        os.remove(output)
        return result


if __name__=="__main__":
    dataset = Dataset("ionosphere")
    clusterer = MyMPCKMeansJava()
    result = clusterer.fit(dataset.data, [],[], dataset.number_of_classes())
