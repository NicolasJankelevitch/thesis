import time


class COBRASLogger:
    def __init__(self):
        # Timing variables
        self.start_time = None
        self.execution_time = None

        #Logging of intermediate results and constraints
        self.intermediate_results = []
        self.all_user_constraints = []

        #Algorithm phases
        self.current_phase = None
        self.algorithm_phases = []

        #Clustering to store
        self.clustering_to_store = None

        #Splitting_data
        self.split_levels = []

        # Predicted unknown constraints
        self.predicted_constraints = []
        self.n_correct_preds = 0

        # RF: Accuracy per number of constraints
        self.accuracy_per_n_constraints = []

    def log_start(self):
        self.start_time = time.time()

    def log_end(self):
        self.execution_time = time.time() - self.start_time

    def log_entering_phase(self, phase):
        self.current_phase = phase

    def update_last_intermediate_result(self, clustering, con_length):
        self.intermediate_results[-1] = (
            clustering.construct_cluster_labeling(), time.time() - self.start_time,
            con_length)

    def get_all_clusterings(self):
        return [cluster for cluster, _, _ in self.intermediate_results]

    def get_runtimes(self):
        return [runtime for _, runtime, _ in self.intermediate_results]

    def get_constraint_lists(self):
        ml = []
        cl = []
        dk = []
        for constraint in self.all_user_constraints:
            if constraint.is_ML():
                ml.append(constraint.get_instance_tuple())
            elif constraint.is_CL():
                cl.append(constraint.get_instance_tuple())
            elif constraint.is_DK():
                dk.append(constraint.get_instance_tuple())
        return ml, cl, dk

    def log_splitlevel(self, splitlevel):
        self.split_levels.append(splitlevel)

    def log_new_user_query(self, constraint,  con_length, clustering_to_store):
        #keep it in all_user_constraints
        self.all_user_constraints.append(constraint)

        #current phases
        self.algorithm_phases.append(self.current_phase)

        # intermediate clustering results
        self.intermediate_results.append(
            (clustering_to_store, time.time() - self.start_time, con_length))