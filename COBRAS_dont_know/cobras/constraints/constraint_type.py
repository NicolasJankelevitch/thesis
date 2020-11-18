from enum import Enum


class ConstraintType(Enum):
    """
        Class to represent the different possible types/answers of a constraint.
            ML: Must-link
            CL: Cannot-link
            DK: Don't know
    """
    ML = 1
    CL = -1
    DK = 0
