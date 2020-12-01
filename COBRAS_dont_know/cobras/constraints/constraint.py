import functools
import json
from .constraint_type import ConstraintType


@functools.total_ordering
class Constraint:
    """
        A class that represents a single constraint
    """

    def __init__(self, i1, i2, const_type: ConstraintType, times_seen=1, times_other_seen=0, purpose="not specified"):
        if i1 == i2:
            raise Exception("Constraint between instance and itself")
        self.i1 = min(i1, i2)
        self.i2 = max(i1, i2)
        self.type = const_type
        self.times_seen = times_seen
        self.times_other_seen = times_other_seen
        self.purpose = purpose

    def add_other_constraint(self, con):
        if not con.get_instance_tuple() == self.get_instance_tuple():
            raise Exception("Trying to add constraint A to a constraint B when A != B ")

        if con.is_ML() == self.is_ML():
            self.times_seen += con.times_seen
            self.times_other_seen += con.times_other_seen
        else:
            self.times_seen += con.times_other_seen
            self.times_other_seen += con.times_seen

    def get_times_seen(self):
        return self.times_seen

    def get_times_other_seen(self):
        return self.times_other_seen

    def contains_instance(self, i):
        return i == self.i1 or i == self.i2

    def has_instance_in_common_with(self, other):
        return self.contains_instance(other.i1) or self.contains_instance(other.i2)

    def get_other_instance(self, i):
        if i == self.i2:
            return self.i1
        elif i == self.i1:
            return self.i2
        raise Exception("get_other_instance with instance that is not part of the constraint")

    def flip(self):
        return Constraint(self.i1, self.i2, ConstraintType.CL if self.is_ML() else ConstraintType.ML, times_seen=self.times_other_seen,
                          times_other_seen=self.times_seen, purpose=self.purpose)

    def to_instance_set(self):
        return {self.i1, self.i2}

    def constraint_type(self) -> ConstraintType:
        return self.type

    def is_ML(self):
        return self.type == ConstraintType.ML

    def is_CL(self):
        return self.type == ConstraintType.CL

    def is_DK(self):
        return self.type == ConstraintType.DK

    def get_instance_tuple(self):
        return self.i1, self.i2

    def to_tuple(self):
        return self.i1, self.i2, self.type

    def __eq__(self, other):
        if other is None:
            return False
        return self.to_tuple() == other.to_tuple()

    def __lt__(self, other):
        return self.to_tuple() < other.to_tuple()

    def __hash__(self):
        return hash((self.i1, self.i2, self.type))

    def full_str(self):
        return str(self) + " times_seen=" + str(self.get_times_seen()) + " times_other=" + str(
            self.get_times_other_seen()) + " for " + self.purpose

    def __repr__(self):
        return self.__str__()

    def to_save_str(self):
        return json.dumps((self.i1, self.i2, self.constraint_type(), self.times_seen, self.times_other_seen, self.purpose))

    @staticmethod
    def create_from_str(string):
        loaded = json.loads(string)
        if loaded is None:
            return None
        i1, i2, cons_type, times_seen, times_other_seen, purpose = loaded
        return Constraint(i1, i2, cons_type, times_seen, times_other_seen, purpose)

    def __str__(self):
        constraint_type = "ML" if self.is_ML() else "CL" if self.is_CL() else "DK"
        return constraint_type + "(" + str(self.i1) + "," + str(self.i2) + ") " + self.purpose
