from enum import Enum
from clustering_algorithms.clustering_algorithms import KMeansClusterAlgorithm, ClusterAlgorithm
from clustering_algorithms.kmeans_fixed_representative import KmeansFixedRepresentative
from clustering_algorithms.kmeans_plus_fixed_representative import KmeansPlusFixedRepresentative
from clustering_algorithms.kmedoids_fixed_representative import KmedoidsFixedRepresentative
from cobras.clusters.cluster import Cluster
from cobras.clusters.clustering import Clustering
from cobras.cobras_logger import COBRASLogger
from cobras.constraints.constraint import Constraint
from cobras.constraints.constraint_index import ConstraintIndex
from cobras.constraints.constraint_type import ConstraintType
from cobras.querier.querier import Querier
from cobras.super_instances.superinstance import SuperInstance, SuperInstanceBuilder
from cobras.super_instances.superinstance_kmeans import KMeans_SuperinstanceBuilder
from heuristics.select_super_instance_heuristics import *
from heuristics.constraint_similarity_euclidean import get_dissimilarity
from heuristics.splitlevel_estimation_strategy import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
import copy


class SplitResult(Enum):
    SUCCESS = 1
    NO_SPLIT_POSSIBLE = 2
    SPLIT_FAILED = 3


class COBRAS:
    def __init__(self,
                 similarity_pred: bool = False,
                 randomforest_pred: bool = False,
                 cobras_plus: bool = False,
                 select_next_query: bool = False,
                 intra_pred: bool = False,
                 max_dist: bool = True,
                 posterior_inter_pred=False,
                 cluster_algo: ClusterAlgorithm = KMeansClusterAlgorithm(),
                 superinstance_builder: SuperInstanceBuilder = KMeans_SuperinstanceBuilder(),
                 split_superinstance_selection_heur: Heuristic = SelectMostInstancesHeuristic(),
                 splitlevel_strategy: SplitLevelEstimationStrategy = StandardSplitLevelEstimationStrategy(
                     SelectMostInstancesHeuristic()),
                 similarity_treshold: int = 0.75):

        # Set seed
        # np.random.seed(2020)
        # data: Dataset without labels
        self.data = None
        # train_indices: Indices of data used for training (Holdout strategy)
        self.train_indices = None
        # querier: Querier object (Querier class instance)
        self.querier = None

        # Cosine similarity prediction
        self.similarity_pred = similarity_pred
        # Random Forest prediction
        self.randomforest_pred = randomforest_pred
        # COBRAS+
        self.cobras_plus = cobras_plus
        # select next query in case of dont know
        self.select_next_query = select_next_query
        self.intra_pred = intra_pred  # if true, DK prediction intra_cluster, if false: DK pred inter_cluster
        self.max_dist = max_dist
        self.posterior_inter_pred = posterior_inter_pred

        # init cobras_cluster_algo
        self.cluster_algo = cluster_algo
        self.superinstance_builder = superinstance_builder

        # init split superinstance selection heuristic
        self.split_superinstance_selection_heur = split_superinstance_selection_heur

        # init splitlevel_heuristic
        self.splitlevel_strategy = splitlevel_strategy

        # variables used during execution
        self.clustering_to_store = None
        self.clustering = None

        # Archive of already answered constraints
        self.constraint_index = ConstraintIndex()

        # Logger object
        self.logger = None

        self.similarity_treshold = similarity_treshold

    def get_constraint_length(self):
        return self.constraint_index.get_number_of_constraints()

    def create_superinstance(self, indices, parent=None) -> SuperInstance:
        return self.superinstance_builder.makeSuperInstance(self.data, indices, self.train_indices, parent)

    def fit(self, X, nb_clusters, train_indices, querier: Querier, logger=None):
        """
            Perform clustering.
            The number of clusters (nb_clusters) is not used in COBRAS but is added as a parameter to have a consistent
            interface over all clustering algorithms
        """

        self.logger = COBRASLogger() if logger is None else logger
        self.logger.log_start()
        self.data = X
        self.train_indices = train_indices if train_indices is not None else range(len(X))
        self.split_superinstance_selection_heur.set_clusterer(self)
        self.splitlevel_strategy.set_clusterer(self)
        self.querier = querier

        # initial clustering: all instances in one superinstance in one cluster
        initial_superinstance = self.create_superinstance(list(range(self.data.shape[0])))
        initial_clustering = Clustering([Cluster([initial_superinstance])])
        self.clustering = initial_clustering

        # last valid clustering keeps the last completely merged clustering
        last_valid_clustering = None

        while not self.querier.query_limit_reached():

            # during this iteration store the current clustering
            self.clustering_to_store = self.clustering.construct_cluster_labeling()

            # splitting phase
            self.logger.log_entering_phase("splitting")
            statuscode = self.split_next_superinstance()
            if statuscode == SplitResult.NO_SPLIT_POSSIBLE:
                # there is no split left to be done
                # we have produced the best clustering
                break
            elif statuscode == SplitResult.SPLIT_FAILED:
                # tried to split a superinstance but failed to split it
                # this is recorded in the superinstance
                # we will split another superinstance in the next iteration
                continue

            # merging phase
            self.logger.log_entering_phase("merging")
            fully_merged = self.merge_containing_clusters(self.clustering)

            # correctly log intermediate results
            if fully_merged:
                self.logger.update_last_intermediate_result(self.clustering, self.get_constraint_length())

            # fill in the last_valid_clustering whenever appropriate
            if fully_merged or last_valid_clustering is None:
                last_valid_clustering = copy.deepcopy(self.clustering)

        self.clustering = last_valid_clustering
        self.logger.log_end()
        all_clusters = self.logger.get_all_clusterings()
        runtimes = self.logger.get_runtimes()
        ml, cl, dk = self.logger.get_constraint_lists()
        return all_clusters, runtimes, ml, cl, dk

    # region SPLITTING

    def split_next_superinstance(self):
        # identify the next super-instance to split
        to_split, originating_cluster = self.identify_superinstance_to_split()
        if to_split is None:
            self.logger.max_split_reached = self.logger.max_split_reached + 1
            return SplitResult.NO_SPLIT_POSSIBLE

        # remove to_split from the clustering
        originating_cluster.super_instances.remove(to_split)
        if len(originating_cluster.super_instances) == 0:
            self.clustering.clusters.remove(originating_cluster)

        # split to_split into new clusters
        split_level = self.determine_split_level(to_split)
        new_super_instances = self.split_superinstance(to_split, split_level)
        #print(len(new_super_instances), " - ", split_level)
        new_clusters = self.add_new_clusters_from_split(new_super_instances)

        if not new_clusters:
            # it is possible that splitting a super-instance does not lead to a new cluster:
            # e.g. a super-instance constains 2 points, of which one is in the test_cobras set
            # in this case, the super-instance can be split into two new ones, but these will be joined
            # again immediately, as we cannot have super-instances containing only test_cobras points (these cannot be
            # queried)
            # this case handles this, we simply add the super-instance back to its originating cluster,
            # and set the already_tried flag to make sure we do not keep trying to split this superinstance

            originating_cluster.super_instances.append(to_split)
            to_split.tried_splitting = True
            to_split.children = None

            if originating_cluster not in self.clustering.clusters:
                self.clustering.clusters.append(originating_cluster)
            return SplitResult.SPLIT_FAILED
        else:
            self.clustering.clusters.extend(new_clusters)

        return SplitResult.SUCCESS

    def identify_superinstance_to_split(self):
        '''
        :return: (the super instance to split, the cluster from which the super instance originates)
        '''
        # if there is only one superinstance return that superinstance as superinstance to split
        if len(self.clustering.clusters) == 1 and len(self.clustering.clusters[0].super_instances) == 1:
            return self.clustering.clusters[0].super_instances[0], self.clustering.clusters[0]

        options = []
        for cluster in self.clustering.clusters:
            if cluster.is_pure:
                #print("IS PURE")
                continue
            if cluster.is_finished:
                #print("IS FINISHED")
                continue
            for superinstance in cluster.super_instances:
                if superinstance.tried_splitting:
                    #print(self.data[superinstance.indices])
                    #("TRIED SPLITTING")
                    continue
                if len(superinstance.indices) == 1:
                    continue
                if len(superinstance.train_indices) < 2:
                    continue
                else:
                    options.append(superinstance)
        if len(options) == 0:
            return None, None
        superinstance_to_split = self.split_superinstance_selection_heur.choose_superinstance(options)
        originating_cluster = \
            [cluster for cluster in self.clustering.clusters if superinstance_to_split in cluster.super_instances][0]

        if superinstance_to_split is None:
            return None, None

        return superinstance_to_split, originating_cluster

    def determine_split_level(self, superinstance):
        # need to make a 'deep copy' here, we will split this one a few times just to determine an appropriate splitting
        # level
        splitlevel = self.splitlevel_strategy.estimate_splitting_level(superinstance)
        self.logger.log_splitlevel(splitlevel)
        return splitlevel

    def split_superinstance(self, si, k, use_basic_kmeans=False):
        clusters = None
        if use_basic_kmeans:
            clusterer = KMeansClusterAlgorithm()
            clusters = clusterer.cluster(self.data, si.indices, k, [], [])
        else:
            if isinstance(self.cluster_algo, KmedoidsFixedRepresentative) or \
                    isinstance(self.cluster_algo, KmeansFixedRepresentative) or \
                    isinstance(self.cluster_algo, KmeansPlusFixedRepresentative):
                self.cluster_algo.parent_repr_idx = si.representative_idx
            # cluster the instances of the superinstance
            clusters = self.cluster_algo.cluster(self.data, si.indices, k, [], [])
        # based on the resulting clusters make new superinstances
        # superinstances with no training instances are assigned to the closest superinstance with training instances
        training = []
        no_training = []
        for new_si_idx in set(clusters):
            cur_indices = [si.indices[idx] for idx, c in enumerate(clusters) if c == new_si_idx]

            si_train_indices = [x for x in cur_indices if x in self.train_indices]
            if len(si_train_indices) != 0:
                training.append(self.create_superinstance(cur_indices, si))
            else:
                no_training.append((cur_indices, np.mean(self.data[cur_indices, :], axis=0)))

        for indices, centroid in no_training:
            closest_train = min(training,
                                key=lambda x: np.linalg.norm(self.data[x.representative_idx, :] - centroid))
            closest_train.indices.extend(indices)

        si.children = training
        return training

    def add_new_clusters_from_split(self, si):
        new_clusters = []
        for x in si:
            new_clusters.append(Cluster([x]))

        if len(new_clusters) == 1:
            return None
        else:
            return new_clusters

    # endregion

    # region MERGING
    def merge_containing_clusters(self, clustering_to_merge):
        """
            Merges the given clustering based on user constraints
            Reusing is done from the constraint index

            :param clustering_to_merge: the clustering to merge
            :type clustering_to_merge: Clustering

            :return: whether or not the merging phase is fully finished
            :rtype bool
        """
        query_limit_reached = False
        merged = True
        dk_pairs = []
        while merged and not self.querier.query_limit_reached():

            clusters_to_consider = [cluster for cluster in clustering_to_merge.clusters if not cluster.is_finished]

            cluster_pairs = itertools.combinations(clusters_to_consider, 2)
            cluster_pairs = [x for x in cluster_pairs if
                             not self.cannot_link_between_clusters(x[0], x[1])
                             and not self.dont_know_between_clusters(x[0], x[1])]
            cluster_pairs = sorted(cluster_pairs, key=lambda x: x[0].distance_to(x[1]))

            merged = False
            for x, y in cluster_pairs:
                if self.cannot_link_between_clusters(x, y):
                    continue
                if self.dont_know_between_clusters(x, y):
                    continue

                must_link_exists = None
                if self.must_link_between_clusters(x, y):
                    must_link_exists = True

                if self.querier.query_limit_reached():
                    query_limit_reached = True
                    break

                # no reuse!
                if must_link_exists is None:
                    new_constraint = self.get_constraint_between_clusters(x, y, "merging", reuse=True)
                    if new_constraint.is_ML():
                        must_link_exists = True
                    if self.posterior_inter_pred and new_constraint.is_DK():
                        dk_pairs.append((new_constraint, x, y))

                if must_link_exists:
                    x.super_instances.extend(y.super_instances)
                    clustering_to_merge.clusters.remove(y)
                    merged = True
                    if self.posterior_inter_pred:
                        dk_pairs = self.update_dk_list(dk_pairs, x, y)
                    break

        if self.posterior_inter_pred:
            # dk_pairs is a queue with pop(0), append()
            # dk_pairs are not really sorted because of updates
            # Queue best sorteren op een goeie volgorde "wat is goeie volgorde?"
            # note: you can use a dictionary to speed the search up
            # dictionary van superinstance representatives --> constraint vinden die je moet aanpassen
            while len(dk_pairs) > 0:
                c, x, y = dk_pairs.pop(0)  # the first element of the queue
                #s1, s2 = self.get_sis(c, x, y)
                dk = True
                while dk:
                    new_constr = self.inter_super_instance_prediction(x, y) #, s1, s2)
                    if new_constr is not None:
                        c.type = new_constr.type  # I don't know
                        if new_constr.is_ML():
                            # do the merge
                            x.super_instances.extend(y.super_instances)
                            clustering_to_merge.clusters.remove(y)
                            # update the queue
                            dk_pairs = self.update_dk_list(dk_pairs, x, y)
                            dk = False
                        if new_constr.is_CL():
                            dk = False
                    else:
                        dk = False
        # other idea before merging save super-instances
        # merging phase gedaan, je hebt daarna een aantal don't knows opgelost
        # herstel de opgeslagen super-instances en herstart de merging phase

        # if self.posterior_inter_pred:
        #     updated = True
        #     while updated:
        #         updated = False
        #         for c, x, y in dk_pairs:
        #             s1, s2 = self.get_sis(c, x, y)
        #             new_constr = self.inter_super_instance_prediction(x, y, s1, s2)
        #             if new_constr is not None:
        #                 c.type = new_constr.type
        #                 if new_constr.is_ML():
        #                     x.super_instances.extend(y.super_instances)
        #                     clustering_to_merge.clusters.remove(y)
        #                     dk_pairs = self.update_dk_list(dk_pairs, x, y)
        #                     updated = True
        #                     break
        #                 # TODO: if DK -> ?

        fully_merged = not query_limit_reached and not merged
        self.logger.end_merging_phase()
        return fully_merged

    def update_dk_list(self, dk_pairs, x, y):
        temp_dks = []
        for c, a, b in dk_pairs:
            if a is y:
                if b is not x:
                    temp_dks.append((c, x, b))
                else:
                    c.type = ConstraintType.ML
            elif b is y:
                if a is not x:
                    temp_dks.append((c, a, x))
                else:
                    c.type = ConstraintType.ML
        return temp_dks

    def get_sis(self, constraint, c1, c2):
        si1 = None
        si2 = None
        for s in c1.super_instances:
            if constraint.i1 in s.indices or constraint.i2 in s.indices:
                si1 = s
                break
        for s in c2.super_instances:
            if constraint.i1 in s.indices or constraint.i2 in s.indices:
                si2 = s
                break
        if si1 is None or si2 is None:
            raise Exception("Instance not found")
        return si1, si2

    def cannot_link_between_clusters(self, c1, c2):
        """
            Checks whether or not a cannot-link exists between the given clusters
        """
        reused = self.check_constraint_reuse_clusters(c1, c2)
        if reused is not None:
            return reused.is_CL()

    def must_link_between_clusters(self, c1, c2):
        """
            Checks whether or not a must-link exists between the given clusters
        """
        reused = self.check_constraint_reuse_clusters(c1, c2)
        if reused is not None:
            return reused.is_ML()

    def dont_know_between_clusters(self, c1, c2):
        """
            Checks whether or not a dont-know exists between the given clusters
        """
        reused = self.check_constraint_reuse_clusters(c1, c2)
        if reused is not None:
            return reused.is_DK()

    # endregion

    # region Constraint querying and constraint reuse
    def get_constraint_between_clusters(self, c1: Cluster, c2: Cluster, purpose, reuse=True):
        if reuse:
            reused_constraint = self.check_constraint_reuse_clusters(c1, c2)
            if reused_constraint is not None:
                return reused_constraint
        si1, si2 = c1.get_comparison_points(c2)
        constr = self.get_constraint_between_superinstances(si1, si2, purpose)
        if self.select_next_query and not self.intra_pred and constr.is_DK() and not self.posterior_inter_pred:  # if there is a dk and inter pred is used, do:
            pred_constr = self.inter_super_instance_prediction(c1, c2) #, si1, si2)
            if pred_constr is not None:
                constr.type = pred_constr.type
                # self.logger.log_prediction_pair((constr, pred_constr))
        return constr

    def get_constraint_between_superinstances(self, s1: SuperInstance, s2: SuperInstance, purpose, reuse=True):
        if reuse:
            reused_constraint = self.check_constraint_reuse_between_representatives(s1, s2)
            if reused_constraint is not None:
                return reused_constraint
        constraint = self.get_new_constraint(s1.representative_idx, s2.representative_idx, purpose)

        if self.select_next_query and constraint.is_DK() and not self.querier.query_limit_reached() and self.intra_pred:
            extra_constr = self.intra_super_instance_prediction(s1, s2)
            if extra_constr is not None:
                constraint.type = extra_constr.type

        return constraint

    def get_constraint_between_instances(self, instance1, instance2, purpose, reuse=True):
        reused_constraint = None
        if reuse:
            reused_constraint = self.check_constraint_reuse_between_instances(instance1, instance2)

        if reused_constraint is not None:
            return reused_constraint

        min_instance = min(instance1, instance2)
        max_instance = max(instance1, instance2)
        return self.get_new_constraint(min_instance, max_instance, purpose)

    def check_constraint_reuse_clusters(self, c1: Cluster, c2: Cluster):
        superinstances1 = c1.super_instances
        superinstances2 = c2.super_instances

        for si1, si2 in itertools.product(superinstances1, superinstances2):
            reused_constraint = self.check_constraint_reuse_between_representatives(si1, si2)
            if reused_constraint is not None:
                return reused_constraint

        return None

    def check_constraint_reuse_superinstances(self, si1, si2):
        reused_constraint = self.check_constraint_reuse_between_representatives(si1, si2)
        return reused_constraint

    def check_constraint_reuse_between_representatives(self, si1, si2):
        self.logger.phase_constraints.add((si1.representative_idx, si2.representative_idx))
        return self.check_constraint_reuse_between_instances(si1.representative_idx, si2.representative_idx)

    def check_constraint_reuse_between_instances(self, i1, i2):
        reused_constraint = None
        ml_constraint = Constraint(i1, i2, ConstraintType.ML)
        cl_constraint = Constraint(i1, i2, ConstraintType.CL)
        dk_constraint = Constraint(i1, i2, ConstraintType.DK)
        constraint_index = self.constraint_index

        if ml_constraint in constraint_index:
            reused_constraint = ml_constraint
        elif cl_constraint in constraint_index:
            reused_constraint = cl_constraint
        elif dk_constraint in constraint_index:
            reused_constraint = dk_constraint
        return reused_constraint

    def get_new_constraint(self, instance1, instance2, purpose):
        min_instance = min(instance1, instance2)
        max_instance = max(instance1, instance2)
        constraint_type = None

        # Check if a similar constraint was already answered (if COBRAS+)
        if self.cobras_plus:
            constraint_type = self.find_similar_constraint(min_instance, max_instance)

        # Do a new Query
        if constraint_type is None:
            constraint_type = self.query_querier(min_instance, max_instance, purpose)

        # posterior prediction
        if self.randomforest_pred and constraint_type == ConstraintType.DK:
            constraint_type = self.try_randomforest_prediction_DK(min_instance, max_instance)

        if self.similarity_pred and constraint_type == ConstraintType.DK:
            constraint_type = self.try_similarity_prediction_DK(min_instance, max_instance)

        if constraint_type is None:
            raise Exception("constraint type None after query")

        new_constraint = Constraint(min_instance, max_instance, constraint_type, purpose=purpose)

        self.constraint_index.add_constraint(new_constraint)
        return new_constraint

    def query_querier(self, instance1, instance2, purpose):
        if self.querier.query_limit_reached():
            print("going over query limit! ", self.get_constraint_length())
        constraint_type = self.querier.query(instance1, instance2)

        self.logger.log_new_user_query(Constraint(instance1, instance2, constraint_type, purpose=purpose),
                                       self.get_constraint_length(), self.clustering_to_store)
        return constraint_type

    def find_similar_constraint(self, i1, i2):
        constraint_1 = Constraint(i1, i2, ConstraintType.DK)
        most_similar_constraint = None
        lowest_dissimilarity = self.similarity_treshold

        for constraint_2 in self.constraint_index:
            if constraint_2.type == ConstraintType.ML:
                dissimilarity = get_dissimilarity(constraint_1, constraint_2, self.data)
                if dissimilarity < lowest_dissimilarity:
                    most_similar_constraint = constraint_2
                    lowest_dissimilarity = dissimilarity

        if most_similar_constraint is not None:
            self.logger.log_predicted_constraint(Constraint(i1, i2, most_similar_constraint.type))
            return most_similar_constraint.type
        return None

    # endregion

    # region new_DK
    def intra_super_instance_prediction(self, si1: SuperInstance, si2: SuperInstance):
        if len(si1.indices) == 1 and len(si2.indices) == 1:
            return None
        pair = None
        indice_pairs = [(i1, i2) for i1, i2 in itertools.product(si1.indices, si2.indices)
                        if not (i1 == si1.representative_idx and i2 == si2.representative_idx)]
        indice_pairs = sorted(indice_pairs, key=lambda p: np.linalg.norm(self.data[p[0]] - self.data[p[1]]))
        if self.max_dist:
            pair = indice_pairs[-1]
        else:
            pair = indice_pairs[0]
        if pair is None:
            raise Exception("No pair found")
        c = self.get_constraint_between_instances(pair[0], pair[1], purpose="intra_pred")
        self.logger.log_extra_constraint(c)
        return c

    def inter_super_instance_prediction(self, c1: Cluster, c2: Cluster):  # , si1: SuperInstance, si2: SuperInstance):
        if not self.querier.query_limit_reached():
            index = -1
            if not self.max_dist:
                index = 0
            #if len(c1.super_instances) == 1 and len(c2.super_instances) == 1:
             #   return None
            super_instance_pairs = [(s1, s2) for s1, s2 in itertools.product(c1.super_instances, c2.super_instances)]
            # if not ((s1 == si1 and s2 == si2) or (s1 == si2 and s2 == si1))]
            super_instance_pairs = sorted(super_instance_pairs, key=lambda x: np.linalg.norm(
                self.data[x[0].representative_idx] - self.data[x[1].representative_idx]))
            pair = None
            pair_found = False
            while not pair_found:
                if len(super_instance_pairs) == 0:
                    break
                pair = super_instance_pairs[index]
                constraint = self.check_constraint_reuse_between_instances(pair[0].representative_idx,
                                                                           pair[1].representative_idx)
                if constraint is None or not constraint.is_DK():
                    pair_found = True
                else:
                    super_instance_pairs.pop(index)
            if not pair_found:
                return None
            constr = self.get_constraint_between_superinstances(pair[0], pair[1], purpose="inter_si")
            self.logger.log_extra_constraint(constr)

            return constr

    # endregion

    # region old_DK
    def try_similarity_prediction_DK(self, A, B):
        cos_threshold = 0.95
        norm_factor_theshold = 0.95  # Minimun percentage
        # Take all existing constraints for both instances
        # AB -> Unknown constraint
        # AC -> Known constraint
        cons_a = self.constraint_index.find_constraints_for_instance(A)
        for cons in cons_a:
            if not cons.is_DK():
                AB = (self.data[B] - self.data[A]).reshape(1, -1)
                C = cons.get_other_instance(A)
                AC = (self.data[C] - self.data[A]).reshape(1, -1)

                cos_sim = cosine_similarity(AB, AC)
                AB_norm = np.linalg.norm(AB)
                AC_norm = np.linalg.norm(AC)
                norm_factor = min(AB_norm, AC_norm) / max(AB_norm, AC_norm)

                if cos_sim > cos_threshold and norm_factor >= norm_factor_theshold:
                    if cons.is_ML() and (cos_sim * AC_norm) > AB_norm:
                        self.logger.predicted_constraints.append(
                            (Constraint(A, B, ConstraintType.ML, purpose="AB_ML"), cons, float(cos_sim)))
                        if self.querier.labels[A] == self.querier.labels[B]:
                            self.logger.n_correct_preds += 1
                        return ConstraintType.ML
                    elif cons.is_CL() and (cos_sim * AB_norm) > AC_norm:
                        self.logger.predicted_constraints.append(
                            (Constraint(A, B, ConstraintType.CL, purpose="AB_CL"), cons, float(cos_sim)))
                        if self.querier.labels[A] != self.querier.labels[B]:
                            self.logger.n_correct_preds += 1
                        return ConstraintType.CL
        # Reverse way
        # BA -> Unknown constraint
        # BC -> Known constraint
        cons_b = self.constraint_index.find_constraints_for_instance(B)
        for cons in cons_b:
            if not cons.is_DK():
                BA = (self.data[A] - self.data[B]).reshape(1, -1)
                C = cons.get_other_instance(B)
                BC = (self.data[C] - self.data[B]).reshape(1, -1)

                cos_sim = cosine_similarity(BA, BC)
                BA_norm = np.linalg.norm(BA)
                BC_norm = np.linalg.norm(BC)
                norm_factor = min(BA_norm, BC_norm) / max(BA_norm, BC_norm)

                if cos_sim > cos_threshold and norm_factor >= norm_factor_theshold:
                    if cons.is_ML() and (cos_sim * BC_norm) > BA_norm:
                        self.logger.predicted_constraints.append(
                            (Constraint(A, B, ConstraintType.ML, purpose="BA_ML"), cons, float(cos_sim)))
                        if self.querier.labels[A] == self.querier.labels[B]:
                            self.logger.n_correct_preds += 1
                        return ConstraintType.ML
                    elif cons.is_CL() and (cos_sim * BA_norm) > BC_norm:
                        self.logger.predicted_constraints.append(
                            (Constraint(A, B, ConstraintType.CL, purpose="BA_CL"), cons, float(cos_sim)))
                        if self.querier.labels[A] != self.querier.labels[B]:
                            self.logger.n_correct_preds += 1
                        return ConstraintType.CL
        return ConstraintType.DK

    def try_randomforest_prediction_DK(self, A, B):
        conf_threshold = 0.9
        n_train_cons_threshold = 60

        if self.querier.queries_asked < n_train_cons_threshold:
            return ConstraintType.DK

        # Take all existing constraints
        constraints = [x for x in self.constraint_index.constraints if not x.is_DK()]

        # Build dataset to train (x1, x2, |x1 - x2|) -> ConstraintType
        if len(constraints) > 0:
            X = []
            y = []
            for cons in constraints:
                i1, i2 = cons.get_instance_tuple()
                cons_norm = np.linalg.norm((self.data[i2] - self.data[i1]).reshape(1, -1)).flatten()
                X.append(np.concatenate((self.data[i1], self.data[i2], cons_norm)))
                y.append(cons.constraint_type().value)

            # Train random forest
            rf = RandomForestClassifier()
            rf.fit(X, y)
            # Predict (A, B, |A-B|)
            cons_norm = np.linalg.norm((self.data[B] - self.data[A]).reshape(1, -1)).flatten()
            X_pred = np.concatenate((self.data[A], self.data[B], cons_norm)).reshape(1, -1)

            y_pred = rf.predict(X_pred)
            y_prob = rf.predict_proba(X_pred)
            if np.any(y_prob >= conf_threshold):
                self.logger.predicted_constraints.append(
                    (Constraint(A, B, ConstraintType(y_pred), purpose="RF")))
                if self.querier.labels[A] == self.querier.labels[B] and ConstraintType(
                        y_pred) == ConstraintType.ML:  # HIT
                    self.logger.n_correct_preds += 1
                    self.logger.accuracy_per_n_constraints.append([self.querier.queries_asked, 1])
                elif self.querier.labels[A] != self.querier.labels[B] and ConstraintType(
                        y_pred) == ConstraintType.CL:  # HIT
                    self.logger.n_correct_preds += 1
                    self.logger.accuracy_per_n_constraints.append([self.querier.queries_asked, 1])
                else:  # MISS
                    self.logger.accuracy_per_n_constraints.append([self.querier.queries_asked, 0])
                return ConstraintType(y_pred)
            else:
                return ConstraintType.DK
        else:
            return ConstraintType.DK
    # endregion
