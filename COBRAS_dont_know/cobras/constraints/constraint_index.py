from collections import defaultdict


class ConstraintIndex:
    """
        A class that stores constraints in a dictionary for fast retrieval
    """

    def __init__(self):
        self.constraint_index = defaultdict(set)
        self.constraints = set()

    def get_all_mls(self):
        return [con for con in self.constraints if con.is_ML()]

    def get_all_cls(self):
        return [con for con in self.constraints if con.is_CL()]

    def get_all_dks(self):
        return [con for con in self.constraints if con.is_DK()]

    def get_number_of_constraints(self):
        return sum(con.get_times_seen() + con.get_times_other_seen() for con in self.constraints)

    def __add_constraint_to_index(self, constraint):
        self.constraints.add(constraint)
        self.constraint_index[constraint.i1].add(constraint)
        self.constraint_index[constraint.i2].add(constraint)

    def add_constraint(self, constraint):
        set_to_search = self.find_constraints_between_instances(constraint.i1, constraint.i2)
        if len(set_to_search) == 0:
            self.__add_constraint_to_index(constraint)
            return True
        elif len(set_to_search) == 1:
            existing_constraint = next(set_to_search.__iter__())
            existing_constraint.add_other_constraint(constraint)
            return False
        else:
            raise Exception("at least two times the same constraint in constraintIndex!")

    def __len__(self):
        return len(self.constraints)

    def __iter__(self):
        return self.constraints.__iter__()

    def __contains__(self, constraint):
        return constraint in self.constraints

    def find_constraints_for_instance(self, instance):
        return self.constraint_index[instance]

    def does_constraint_between_instances_exist(self, i1, i2):
        return len(self.find_constraints_between_instances(i1, i2)) > 0

    def get_constraint(self, constraint):
        if constraint not in self:
            return None
        return next(con for con in self.find_constraints_between_instances(constraint.i1, constraint.i2) if
                    con.is_ML() == constraint.is_ML())

    def find_constraints_between_instances(self, i1, i2):
        return self.constraint_index[i1].intersection(self.constraint_index[i2])
