from cobras.constraints.constraint import Constraint
from scipy.spatial import distance


def get_dissimilarity(c1: Constraint, c2: Constraint, data):
    print(data[c1.i1])
    length1 = distance.euclidean(data[c1.i1], data[c1.i2])
    length2 = distance.euclidean(data[c2.i1], data[c2.i2])
    denominator = (length1 + length2)/2
    distance1 = distance.euclidean(data[c1.i1], data[c2.i1])
    distance2 = distance.euclidean(data[c1.i2], data[c2.i2])
    distance3 = distance.euclidean(data[c1.i1], data[c2.i2])
    distance4 = distance.euclidean(data[c1.i2], data[c2.i1])
    numerator = min(max(distance1, distance2), max(distance3, distance4))
    return numerator / denominator
