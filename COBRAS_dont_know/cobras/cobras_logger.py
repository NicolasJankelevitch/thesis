import time

from cobras.constraints.constraint_type import ConstraintType


class COBRASLogger:
    def __init__(self):
        # Timing variables
        self.start_time = None
        self.execution_time = None

        #Logging of intermediate results and constraints
        self.intermediate_results = []
        self.queried_constraints = []
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

        # amount of reused constraints
        self.reused_constraints = []
        self.phase_constraints = set(())
        self.constraints_previously_used = set(())

        # check max split reached
        self.max_split_reached = 0

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
        for li in [self.queried_constraints]:#, self.predicted_constraints]:
            for constraint in li:
                if isinstance(constraint,tuple):
                    constraint = constraint[0]
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
        self.queried_constraints.append(constraint)
        #current phases
        self.algorithm_phases.append(self.current_phase)

        # intermediate clustering results
        self.intermediate_results.append(
            (clustering_to_store, time.time() - self.start_time, con_length))

    def end_merging_phase(self):
        self.reused_constraints.extend(self.phase_constraints.intersection(self.constraints_previously_used))
        self.constraints_previously_used.update(self.phase_constraints)
        self.phase_constraints = set(())

    def log_predicted_constraint(self, constraint):
        self.predicted_constraints.append(constraint)

    def log_extra_constraint(self, constr): #item 1 in pair is original DK constraint, item 2 is asked used to predict 1
        ctype = None
        if constr.type is ConstraintType.ML:
            ctype = "ML"
        elif constr.type is ConstraintType.CL:
            ctype = "CL"
        else:
            ctype = "DK"
        self.predicted_constraints.append((constr.i1, constr.i2, ctype))
        #self.extra_asked.append((pair[1].i1, pair[1].i2, ctype))
