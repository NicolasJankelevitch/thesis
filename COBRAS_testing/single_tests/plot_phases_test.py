import logging
import sys
import time
from logging import Logger

import matplotlib.pyplot as plt
from adjustText import adjust_text

from cobras_ts.cobras import COBRAS
from cobras_ts.cobras_logger import ClusteringLogger
from datasets import Dataset
from evaluate_clusterings.evaluation.ari import get_ARI
from generate_clusterings.algorithms.my_mpckmeans import MyMPCKMeans
from generate_clusterings.algorithms.my_npu import NPU
from generate_clusterings.algorithms.noise_robust_npu import NoiseRobustNPU
from querier.labelquerier import LabelQuerier
from querier.noisy_labelquerier import ProbabilisticNoisyQuerier

counter = 0


def plotting(logger, ground_truth_querier):
    # use latex for font rendering
    # mpl.rcParams['text.usetex'] = True
    global counter
    # plt.ion()
    fig, phase_plot = plt.subplots(1, 1, sharex='all', figsize=(18, 8))
    # ari_plot.plot(ari)

    # all the user constraints that have been received
    all_user_constraint = logger.all_user_constraints
    all_noisy_user_constraints = [(idx, con) for idx, con in enumerate(all_user_constraint) if
                                  ground_truth_querier._query_points(*con.get_instance_tuple()) != con.is_ML()]

    # the constraints that are detected as noisy
    detected_noisy_data = logger.detected_noisy_constraint_data

    phase_data = logger.algorithm_phases

    phase_plot.plot(phase_data)

    texts = []
    # plot constraint reusage
    # for x, (correctly_reused, wrongly_reused, _) in enumerate(logger.constraint_reusing_data):
    #     if correctly_reused > 0:
    #         phase_plot.scatter([x], [phase_data[x]], marker='o', c='tab:blue', s=40)
    #     if wrongly_reused > 0:
    #         phase_plot.scatter([x], [phase_data[x]], marker='o', c='tab:red', s=30)

    for (x, detected_noisy_constraint) in detected_noisy_data:

        if x < len(phase_data):

            # print(detected_noise, constraint_type, i1, i2)

            real_constraint_type = ground_truth_querier._query_points(detected_noisy_constraint.i1,
                                                                      detected_noisy_constraint.i2)
            phase_plot.scatter([x], [phase_data[x]], marker='D', c='g', s=90)
            if real_constraint_type != detected_noisy_constraint.is_ML():  # the detected noise was indeed noisy
                text = phase_plot.text(x, phase_data[x], str(detected_noisy_constraint), color="green", ha="center",
                                       va="center")
            else:
                text = phase_plot.text(x, phase_data[x], str(detected_noisy_constraint), color="tab:orange",
                                       ha="center",
                                       va="center")
            texts.append(text)
    for idx, noisy_constraint in all_noisy_user_constraints:
        constraint_str = str(noisy_constraint)
        phase_plot.scatter([idx], [phase_data[idx]], marker='x', c='r', s=50)
        text = phase_plot.text(idx, phase_data[idx], constraint_str, color='red', ha='center', va='center')
        texts.append(text)
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black'), only_move={'points': 'y', 'text': 'y'})
    phase_plot.set_xticks(range(0, 61, 1), [str(n) if n % 5 == 0 else "" for n in range(0, 61)])
    plt.ylabel("Algorithm phase")
    plt.xlabel("#queries")
    plt.tight_layout()
    plt.show()
    counter += 1

    # plt.waitforbuttonpress()

    plt.close(fig)


def simpleRun(dataset_name):
    start = time.time()
    dataset = Dataset(dataset_name)

    basic_logger = Logger("cobras", logging.DEBUG)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    basic_logger.addHandler(handler)

    cobras_logger = ClusteringLogger()
    ALGO_TO_USE = "cobras"
    if ALGO_TO_USE == "cobras":
        clusterer = COBRAS(minimum_approximation_order=3, maximum_approximation_order=5, noise_probability=0.05,
                           certainty_threshold=0.95, correct_noise=True,
                           logger=basic_logger, cobras_logger=cobras_logger)

        # logger.vis = ConstraintGraphVisualisation(clusterer)
        clusterings, _, _, _ = clusterer.fit(dataset.data, -1, None,
                                             ProbabilisticNoisyQuerier(cobras_logger, dataset.target, 0.05, 100))

        logger = clusterer._cobras_log

    elif ALGO_TO_USE == 'nnpu-mpck':
        clusterer = NoiseRobustNPU(MyMPCKMeans(), 3, 8, 0.05, 0.95, cobras_logger)
        clusterings, _, _, _ = clusterer.fit(dataset.data, dataset.number_of_classes(), None,
                                             ProbabilisticNoisyQuerier(cobras_logger, dataset.target, 0.05, 50))
        logger = clusterer.logger

    elif ALGO_TO_USE == 'npu-mpck':
        clusterer = NPU(MyMPCKMeans())
        clusterings, _,_,_ = clusterer.fit(dataset.data, dataset.number_of_classes(), None, ProbabilisticNoisyQuerier(None, dataset.target, 0.05,50))
        logger = None
    else:
        raise Exception(f"unknown algorithm to use {ALGO_TO_USE}")
    # print(clustering, dataset.target)
    end = time.time()
    ground_truth_querier =LabelQuerier(None, dataset.target, None)
    print("took: ", end - start)
    # print("correctly reused", sum([x for x, _, _ in logger.constraint_reusing_data]))
    # print("wrongly reused", sum([x for _, x, _ in logger.constraint_reusing_data]))
    print("produced ARI:", get_ARI(clusterings[-1], dataset.target))
    aris = [get_ARI(cluster, dataset.target) for cluster in clusterings]
    fig, ax = plt.subplots(1,1)
    ax.plot(aris)
    fig.show()
    if logger is not None:
        print("corrected_constraint_set + mistakes:")
        # logger.add_mistake_information(ground_truth_querier)
        # for constraint_number, _, mistakes in logger.corrected_constraint_sets:
        #     print("\t", constraint_number, ")", len(mistakes), "mistakes ", mistakes)
        plotting(logger, ground_truth_querier)


if __name__ == '__main__':
    # while True:
    simpleRun("iris")
