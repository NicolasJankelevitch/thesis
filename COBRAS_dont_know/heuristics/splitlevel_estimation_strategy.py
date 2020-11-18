from abc import ABC, abstractmethod


class SplitLevelEstimationStrategy(ABC):
    def __init__(self):
        self.cobras_clusterer = None

    @abstractmethod
    def estimate_splitting_level(self, superinstance):
        pass

    @abstractmethod
    def get_name(self):
        pass

    def set_clusterer(self, cobras_clusterer):
        self.cobras_clusterer = cobras_clusterer


class ConstantSplitLevelEstimationStrategy(SplitLevelEstimationStrategy):
    def __init__(self, constant_split_level):
        super().__init__()
        self.constant_split_level = constant_split_level

    def estimate_splitting_level(self, superinstance):
        return min(len(superinstance.train_indices), self.constant_split_level)

    def get_name(self):
        return "ConstantSplittingLevel({})".format(self.constant_split_level)


class StandardSplitLevelEstimationStrategy(SplitLevelEstimationStrategy):
    def __init__(self, superinstance_selection_strategy):
        super().__init__()
        self.superinstance_selection_strategy = superinstance_selection_strategy

    def get_name(self):
        return "StandardSplitLevel({})".format(type(self.superinstance_selection_strategy).__name__)

    def estimate_splitting_level(self, superinstance):
        max_split = len(superinstance.indices)

        si_copy = superinstance.copy()
        must_link_found = False
        split_level = 0
        while not must_link_found and not self.cobras_clusterer.querier.query_limit_reached():
            new_sis = self.cobras_clusterer.split_superinstance(si_copy, 2)

            if len(new_sis) == 1:
                # We cannot split any further along this branch, we reached the splitting level
                break

            s1 = new_sis[0]
            s2 = new_sis[1]
            if self.cobras_clusterer.get_constraint_between_superinstances(s1, s2, "determine splitlevel").is_ML():
                must_link_found = True
                continue
            else:
                # The constraint is a cannot link or a dont know answer
                split_level += 1

            si_to_choose = []
            if len(s1.train_indices) >= 2:
                si_to_choose.append(s1)
            if len(s2.train_indices) >= 2:
                si_to_choose.append(s2)

            if len(si_to_choose) == 0:
                # Neither of the superinstances have enough training instances
                break

            # Continue with the superinstance chosen by the heuristic
            si_copy = self.superinstance_selection_strategy.choose_superinstance(si_to_choose)

        split_level = max(split_level, 1)
        split_n = 2 ** int(split_level)
        return min(max_split, split_n)
