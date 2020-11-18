

import jsonpickle as jsonpickle
import numpy as np
from datasets import Dataset
import time
from evaluate_clusterings.evaluation.ari import get_ARI
def calculate_satisfaction_stats(clustering, ml, cl):
    sat_ml, unsat_ml = [], []
    for i1, i2 in ml:
        if clustering[i1]== clustering[i2]:
            sat_ml.append((i1,i2))
        else:
            unsat_ml.append((i1,i2))
    sat_cl, unsat_cl = [], []
    for i1, i2 in cl:
        if clustering[i1] != clustering[i2]:
            sat_cl.append((i1, i2))
        else:
            unsat_cl.append((i1, i2))
    return set(sat_ml), set(unsat_ml), set(sat_cl), set(unsat_cl)



def run_semi_supervised_algo(dataset:Dataset, semi_supervised_clusterer, ml, cl, noisy_ml, noisy_cl, result_path):
    data = dataset.data
    nb_classes = dataset.number_of_classes()
    start_time = time.process_time()
    clustering = semi_supervised_clusterer.fit(data, ml, cl, nb_classes)
    end_time =time.process_time()
    execution_time = end_time - start_time
    sat_ml, unsat_ml, sat_cl, unsat_cl = calculate_satisfaction_stats(clustering, ml, cl)
    # detected as noisy but not noisy
    false_positives = unsat_ml.difference(noisy_ml).union(unsat_cl.difference(noisy_cl))
    # detected as noisy and really noisy
    true_positives = unsat_ml.intersection(noisy_ml).union(unsat_cl.intersection(noisy_cl))
    # detected as normal and really normal
    true_negatives = sat_ml.difference(noisy_ml).union(sat_cl.difference(noisy_cl))
    # detected as normal but not normal
    false_negatives = sat_ml.intersection(noisy_ml).union(sat_cl.intersection(noisy_cl))
    # ari
    if len(false_positives) == 0 and len(true_positives) == 0:
        precision = None
    else:
        precision = len(true_positives) / (len(false_positives) + len(true_positives))
    if len(true_positives) + len(false_negatives) == 0:
        recall = None
    else:
        recall = len(true_positives) / (len(true_positives) + len(false_negatives))
    ari = get_ARI(clustering, dataset.target)
    result_dict = {
        "execution_time":execution_time,
        "ari":ari,
        "sat_ml": sat_ml,
        "unsat_ml":unsat_ml,
        "sat_cl":sat_cl,
        "unsat_cl":unsat_cl,
        "false_positives":false_positives,
        "false_negatives":false_negatives,
        "true_positives":true_positives,
        "true_negatives": true_negatives,
        "precision": precision,
        "recall": recall,
    }
    with result_path.open(mode='w') as output_file:
        string = jsonpickle.encode(result_dict)
        output_file.write(string)
