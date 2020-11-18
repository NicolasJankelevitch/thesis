from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import math
import time
import random

from generate_clusterings.algorithms.my_cosc import NoCOSCResult
from querier.querier import MaximumQueriesExceeded
from config import COSC_PATH
class NPU:
    def __init__(self, clusterer=None, debug = False):
        self.clusterer = clusterer
        self.debug = debug
        self.got_no_cosc_result = []
        self.raw_run_name = None

    @property
    def dataset_name(self):
        if self.raw_run_name is None:
            return "UNKOWN"
        else:
            return Path(self.raw_run_name).parent.name

    @property
    def run_name(self):
        if self.raw_run_name is None:
            return "UNKOWN"
        else:
            return Path(self.raw_run_name).name

    def fit(self, X, nb_clusters, train_indices, querier):
        if train_indices is None:
            train_indices = list(range(0,len(X)))
        # A bit of a hack to calculate W for COSC


        mls = []
        cls = []
        all_clusts = []
        runtimes = []

        ml = []
        cl = []

        start = time.time()
        if hasattr(self.clusterer, "signal_start"):
            self.clusterer.signal_start(X)
        print("started after ", time.time() - start)

        hoods = [[random.choice(train_indices)]]
        t = 0

        query_limit_reached = False
        while not query_limit_reached:
            # print(len(ml) + len(cl))
            pts_queried_in_iteration = 0
            try:
                pred = self.clusterer.fit(X, ml, cl, nb_clusters)
                if self.debug:
                    print(f"RESULT {len(ml) + len(cl)}, {self.dataset_name}, {self.run_name}")
            except NoCOSCResult:
                self.got_no_cosc_result.append(len(ml)+len(cl))
                # just use the previous pred]
                if self.debug:
                    print(f"NO RESULT {len(ml) + len(cl)}, {self.dataset_name}, {self.run_name}")
                pass

            x_star, hood_probs = self.most_informative(X, pred, hoods, train_indices)
            if x_star is None:
                return all_clusts, runtimes, ml, cl
                # return all_clusts, runtimes, mls, cls
            if not 0 <= x_star < X.shape[0]:
                print(
                    "invalid x_star!-----------------------------------------------------------------------------------------------------------------",
                    x_star)

            hoods_decreasing_prob = (-hood_probs).argsort()
            ml_achieved = False
            for hood in hoods_decreasing_prob:
                pt_to_query = hoods[hood][0]  # take the first point of the hood, doesn't matter

                try:
                    constraint = querier.query(x_star, pt_to_query)
                    pts_queried_in_iteration += 1
                except MaximumQueriesExceeded:

                    last_result = None
                    try:
                        self.clusterer.fit(X, ml, cl, nb_clusters)
                    except NoCOSCResult:
                        pass
                    query_limit_reached = True
                    break

                if constraint.is_ML():
                    ml_achieved = True
                    if x_star < pt_to_query:
                        ml.append([int(x_star), int(pt_to_query)])
                    else:
                        ml.append([int(pt_to_query), int(x_star)])
                    hoods[hood].append(x_star)
                    break
                else:
                    if x_star < pt_to_query:
                        cl.append([int(x_star), int(pt_to_query)])
                    else:
                        cl.append([int(pt_to_query), int(x_star)])



            if isinstance(pred, np.ndarray):
                to_add = pred.tolist()
            else:
                to_add = pred[:]

            for j in range(pts_queried_in_iteration):
                all_clusts.append(to_add[:])
                runtimes.append(time.time() - start)
                mls.append(ml[:])
                cls.append(cl[:])

            if query_limit_reached:
                if last_result is not None:
                    if isinstance(last_result, np.ndarray):
                        to_add = last_result.tolist()
                    else:
                        to_add = last_result[:]
                    all_clusts[-1] = to_add
                break

            if not ml_achieved:
                hoods.append([x_star])

            # for noise robust integration just add a confirm and correct call here!
            # should be easy! 
        if hasattr(self.clusterer, "signal_end"):
            self.clusterer.signal_end()
        # print(len(ml) + len(cl))
        # return all_clusts, runtimes, mls, cls
        return all_clusts, runtimes, ml, cl

    def most_informative(self,X, pred, hoods, train_idx):

        forest = RandomForestClassifier(n_estimators=50)

        forest.fit(X, np.array(pred))

        flat_hoods = [item for sublist in hoods for item in sublist]

        forest_pred = np.array(forest.apply(X))
        sims = np.zeros((X.shape[0], X.shape[0]))

        for i in range(X.shape[0]):
            for j in range(i, X.shape[0]):
                diff = forest_pred[i, :] - forest_pred[j, :]
                agreement = diff.shape[0] - np.count_nonzero(
                    diff)  # number of trees - number of times they reach a different leaf
                sims[i, j] = 1.0 * agreement / forest_pred.shape[1]
                sims[j, i] = sims[i, j]

        sum_pt_in_hood = np.zeros((X.shape[0], len(
            hoods)))  # sum of similarity to calculate probability that point x belongs to neighborhood i
        for x in range(X.shape[0]):
            for i in range(len(hoods)):
                for hood_pt in hoods[i]:
                    sum_pt_in_hood[x, i] += sims[x, hood_pt]

        pr_pt_in_hood = np.zeros((X.shape[0], len(hoods)))
        pt_uncertainty = np.zeros(X.shape[0])
        pt_expectation = np.zeros(X.shape[0])

        pt_informativeness = np.zeros(X.shape[0])

        for x in train_idx:

            if x in flat_hoods:
                pt_informativeness[
                    x] = -1  # could be any negative value, simply to assure that we do not pick the same point twice
                continue

            entropy_sum = 0
            this_x_probs = []
            for i in range(len(hoods)):

                numerator = sum_pt_in_hood[x, i] / len(hoods[i])
                denominator = 0
                for p in range(len(hoods)):
                    denominator += (sum_pt_in_hood[x, p] / len(hoods[p]))

                if denominator == 0:
                    pr_pt_in_hood[x, i] = 1.0 / len(hoods)
                else:
                    pr_pt_in_hood[x, i] = numerator / denominator

                if pr_pt_in_hood[x, i] != 0:
                    entropy_sum += (pr_pt_in_hood[x, i] * math.log(pr_pt_in_hood[x, i]) / math.log(2))

                this_x_probs.append(pr_pt_in_hood[x, i])

            pt_uncertainty[x] = -1.0 * entropy_sum
            pt_expectation[x] = sum([(k + 1) * prob for k, prob in enumerate(sorted(this_x_probs, reverse=True))])

            pt_informativeness[x] = pt_uncertainty[x] / pt_expectation[x]
            # pt_informativeness[x] = pt_uncertainty[x]

        while np.count_nonzero(pt_informativeness != -1.0) > 0:

            mi_pt = np.argmax(pt_informativeness)
            equally_informative = np.where(pt_informativeness == pt_informativeness[mi_pt])[0]

            if len([x for x in equally_informative if x in train_idx]) == 0:
                pt_informativeness[equally_informative] = -1.0
            else:
                break

        if np.count_nonzero(pt_informativeness != -1.0) == 0:
            return None, None
        else:
            mi_pt_r = random.choice([x for x in equally_informative if x in train_idx])
            hood_probs = pr_pt_in_hood[mi_pt_r, :]
            return mi_pt_r, hood_probs
