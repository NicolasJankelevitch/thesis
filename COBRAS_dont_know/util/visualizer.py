import matplotlib.pyplot as plt


def plot_clustering(data, target, name, save = False):
    if data.data.shape[1] != 2:
        raise Exception(
            "plot dataset is only supported for datasets with dimension 2. This dataset {} has dimensionality {}".format(
                name, data.data.shape[1]))
    figure = plt.figure()
    cluster_ids = set(target)
    for cluster_id in cluster_ids:
        cluster_indices = [idx for idx, cluster in enumerate(target) if cluster == cluster_id]
        cluster_points = data.data[cluster_indices]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1])
    plt.title(name)

    if save:
        plt.savefig("/Users/nicol/Documents/KUL 2020-2021/thesis/2d datasets visualized/COBRAS+/"+ name + ".png")
    else:
        plt.show()
    # figure.close()
    plt.close(figure)