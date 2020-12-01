import math
import random

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from cobras_ts.cobras_logger import ClusteringLogger
from noise_robust.datastructures.certainty_constraint_set import NewCertaintyConstraintSet
from noise_robust.noise_robust_possible_worlds import gather_extra_evidence
from querier.querier import MaximumQueriesExceeded


class NoiseRobustNPU:
    def __init__(self, clusterer, min_approx_order, max_approx_order, noise_prob, confidence_threshold,
                 cluster_logger=None, **kwargs):
        self.clusterer = clusterer
        self.min_approx_order = min_approx_order
        self.max_approx_order = max_approx_order
        self.noise_prob = noise_prob
        self.confidence_threshold = confidence_threshold
        self.logger = cluster_logger if cluster_logger is not None else ClusteringLogger()

    def fit(self, X, nb_clusters, train_indices, querier):
        if train_indices is None:
            train_indices = list(range(0, len(X)))
        # A bit of a hack to calculate W for COSC
        calculate_info_function = getattr(self.clusterer, "calculate_cacheable_info", None)
        if callable(calculate_info_function):
            self.clusterer.calculate_cacheable_info(X)

        certainty_constraint_set = NewCertaintyConstraintSet(self.min_approx_order, self.max_approx_order,
                                                             self.noise_prob, self.logger, None)
        constraint_index = certainty_constraint_set.constraint_index

        self.logger.log_start_clustering()
        self.logger.log_entering_phase("NPU")
        self.logger.update_clustering_to_store([])

        hoods = [[random.choice(train_indices)]]
        t = 0

        query_limit_reached = False
        while not query_limit_reached:

            pts_queried_in_iteration = 0
            ml, cl = constraint_index.get_ml_and_cl_tuple_lists()
            pred = self.clusterer.fit(X, ml, cl, nb_clusters)
            self.logger.update_last_intermediate_result(pred)
            self.logger.update_clustering_to_store(pred)
            x_star, hood_probs = self.most_informative(X, pred, hoods, train_indices)

            if x_star is None:
                break
                # return all_clusts, runtimes, mls, cls

            hoods_decreasing_prob = (-hood_probs).argsort()
            ml_achieved = False
            for hood in hoods_decreasing_prob:
                pt_to_query = hoods[hood][0]  # take the first point of the hood, doesn't matter

                t += 1
                pts_queried_in_iteration += 1
                reused_constraints = constraint_index.find_constraints_between_instances(x_star, pt_to_query)
                assert len(reused_constraints) < 2
                if len(reused_constraints) == 1:
                    print("REUSED SOMETHING")
                    con = reused_constraints[0]
                else:
                    try:
                        con = querier.query(x_star, pt_to_query, purpose="NPU")
                        certainty_constraint_set.add_constraint(con)
                    except MaximumQueriesExceeded:
                        query_limit_reached = True
                        break

                if con.is_ML():
                    ml_achieved = True
                    # if x_star < pt_to_query:
                    #     ml.append([int(x_star), int(pt_to_query)])
                    # else:
                    #     ml.append([int(pt_to_query), int(x_star)])
                    hoods[hood].append(x_star)
                    break
                else:
                    pass
                    # if x_star < pt_to_query:
                    #     cl.append([int(x_star), int(pt_to_query)])
                    # else:
                    #     cl.append([int(pt_to_query), int(x_star)])

            if query_limit_reached:
                break
            # if isinstance(pred, np.ndarray):
            #     to_add = pred.tolist()
            # else:
            #     to_add = pred[:]

            # for j in range(pts_queried_in_iteration):
            #     all_clusts.append(to_add[:])
            #     runtimes.append(time.time() - start)
            #     mls.append(ml[:])
            #     cls.append(cl[:])

            if not ml_achieved:
                hoods.append([x_star])

            # confirm and correct the constraints
            try:
                gather_extra_evidence(certainty_constraint_set, certainty_constraint_set.constraint_index.constraints,
                                      self.confidence_threshold, querier, self.logger)
                # # fix the hoods!
                # noise_list = certainty_constraint_set.get_noise_list()
                # component_tracker_to_find = None
                # for noise_item, component_tracker in noise_list:
                #     if noise_item == ():
                #         component_tracker_to_find = component_tracker
                #         break
                # if component_tracker_to_find is None:
                #     print("failed to find correct componenttracker!")
                # else:
                #     hoods = []
                #     for component in component_tracker_to_find.components:
                #         hoods.append(list(component.instances))

                self.logger.log_corrected_constraint_set(certainty_constraint_set.constraint_index.constraints)
                self.logger.log_entering_phase("NPU")
            except MaximumQueriesExceeded:
                break

        # return all_clusts, runtimes, mls, cls
        intermediate_results = self.logger.intermediate_results
        all_clusters = [x for x, _, _ in intermediate_results]
        runtimes = [x for _, x, _ in intermediate_results]
        ml, cl = self.logger.get_ml_cl_constraint_lists()
        return all_clusters, runtimes, ml, cl

    def most_informative(self, X, pred, hoods, train_idx):
        # train a random forest to predict the cluster labels
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
