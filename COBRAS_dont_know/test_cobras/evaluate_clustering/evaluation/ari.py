from sklearn import metrics


def intermediate_results_to_ARIs(intermediate_results, target, train_indices = None):
    aris = []
    prev_clustering = None
    prev_ARI = None
    for clustering in intermediate_results:
        if prev_clustering is None or clustering != prev_clustering:
            ari = get_ARI(clustering, target, train_indices = train_indices)
            aris.append(ari)
            prev_clustering = clustering
            prev_ARI = ari
        else:
            #just reuse ARI of the clustering before
            aris.append(prev_ARI)
    return aris


def get_ARI(cluster, target, train_indices=None, test_indices=None):
    if test_indices is not None:
        return get_ARI_test_indices(cluster, target, test_indices)
    if train_indices is not None:
        test_indices = [x for x in range(target.shape[0]) if x not in train_indices]
        return get_ARI_test_indices(cluster, target, test_indices)
    else:
        return metrics.adjusted_rand_score(cluster, target)


def get_ARI_test_indices(cluster, target, test_indices):
    labels_to_evaluate = [x for i, x in enumerate(target) if i in test_indices]
    # print(cluster)
    to_evaluate = [x for i, x in enumerate(cluster) if i in test_indices]
    return metrics.adjusted_rand_score(labels_to_evaluate, to_evaluate)