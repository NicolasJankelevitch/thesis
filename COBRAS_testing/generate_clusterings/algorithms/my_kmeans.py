from sklearn.cluster import KMeans
class KMeansClusterer:
    def __init__(self):
        pass

    def fit(self, X, ml, cl, nb_clusters):
        clusterer = KMeans(nb_clusters)
        clusterer.fit(X)
        clustering = clusterer.labels_
        return clustering