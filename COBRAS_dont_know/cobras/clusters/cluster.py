import itertools


class Cluster:
    """

    """
    def __init__(self, super_instances):
        self.super_instances = super_instances
        # in the visual querier, the user can indicate that the entire cluster is pure
        self.is_pure = False
        # is set to True whenever splitting the super-instance fails i.e. if there is only one training instance
        self.is_finished = False

    def distance_to(self, other_cluster):
        # calculates the distance between 2 clusters by calculating the distance between the closest pair of super-instances
        super_instance_pairs = itertools.product(self.super_instances, other_cluster.super_instances)
        return min([x[0].distance_to(x[1]) for x in super_instance_pairs])

    """
    def get_comparison_points(self, other_cluster):
        # any super-instance should do, no need to find closest ones!
        return self.super_instances[0], other_cluster.super_instances[0]
    """

    def get_comparison_points(self, other_cluster):
        # Choose the closest super instances of the clusters
        superinstances1 = self.super_instances
        superinstances2 = other_cluster.super_instances

        min_distance = 1
        closest_sis = (superinstances1[0], superinstances2[0])
        for si1, si2 in itertools.product(superinstances1, superinstances2):
            if si1.distance_to(si2) < min_distance:
                min_distance = si1.distance_to(si2)
                closest_sis = (si1, si2)
        return closest_sis

    def get_all_points_per_superinstance(self):
        all_pts = []
        for superinstance in self.super_instances:
            all_pts.append(superinstance.indices)
        return all_pts

    def get_all_points(self):
        all_pts = []
        for super_instance in self.super_instances:
            all_pts.extend(super_instance.indices)
        return all_pts


