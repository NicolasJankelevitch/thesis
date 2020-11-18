import random
import itertools
import numpy as np


class Heuristic:
    def __init__(self):
        self.clusterer = None

    def get_name(self):
        return type(self).__name__

    def set_clusterer(self, clusterer):
        self.clusterer = clusterer

    def choose_superinstance(self, superinstances):
        pass


class SelectRandomHeuristic(Heuristic):
    # Choose a random superinstance
    def choose_superinstance(self, superinstances):
        return random.choice(superinstances)


class SelectLeastInstancesHeuristic(Heuristic):
    # Choose the superinstance with less instances
    def choose_superinstance(self, superinstances):
        return min(superinstances, key=lambda superinstance: len(superinstance.indices))


class SelectMostInstancesHeuristic(Heuristic):
    # Choose the superinstance with more instances
    def choose_superinstance(self, superinstances):
        return max(superinstances, key=lambda superinstance: len(superinstance.indices))


class SelectMaximumMeanDistanceBetweenInstances(Heuristic):
    # Choose the superinstance that is more spread on average
    def choose_superinstance(self, superinstances):
        best_superinstance = None
        best_score = None
        for si in superinstances:
            mean_distance_between_instances = np.mean([np.linalg.norm(instance1 - instance2) for instance1, instance2 in
                                                       itertools.combinations(si.data[si.indices], 2)])
            if best_score is None or mean_distance_between_instances > best_score:
                best_score, best_superinstance = mean_distance_between_instances, si

        return best_superinstance


class SelectMaximumDistanceToRepresentative(Heuristic):
    # Choose the superinstance with the largest radius
    def choose_superinstance(self, superinstances):
        best_superinstance = None
        best_score = None
        for si in superinstances:
            representative = si.data[si.representative_idx]
            max_distance_to_representative = max(np.array(np.linalg.norm(representative - instance))
                                                 for instance in si.data[si.indices])
            if best_score is None or max_distance_to_representative > best_score:
                best_score, best_superinstance = max_distance_to_representative, si

        return best_superinstance


class SelectMaximumMeanDistanceToRepresentative(Heuristic):
    # Choose the superinstance with more distance to the representative on average
    def choose_superinstance(self, superinstances):
        best_superinstance = None
        best_score = None

        for superinstance in superinstances:
            representative = superinstance.data[superinstance.representative_idx]
            mean_distance_to_representative = np.mean([np.linalg.norm(representative - instance) for instance in superinstance.data[superinstance.indices]])
            if best_score is None or mean_distance_to_representative > best_score:
                best_score, best_superinstance= mean_distance_to_representative, superinstance
        return best_superinstance
