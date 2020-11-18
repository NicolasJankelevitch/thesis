from datasets import Dataset
from generate_clusterings.algorithms.my_cosc import MyCOSCMatlab
from generate_clusterings.algorithms.my_npu import NPU
from generate_clusterings.clustering_task import ClusteringTask
from cobras_dont_know.cobras.querier.labelquerier import LabelQuerier


def matlab_test():
    dataset = Dataset("iris")
    clusterer = MyCOSCMatlab()
    clusterer.signal_start(dataset.data)
    result = clusterer.fit(dataset.data, [(1,2),(2,3),(3,dataset.number_of_instances())], [(10,12),(23,16)], dataset.number_of_classes())
    print(result)
    clusterer.signal_end()

def run_npu_COSC():
    dataset = Dataset("iris")

    clusterer = NPU(MyCOSCMatlab(), debug = False)
    clusterer.fit(dataset.data, dataset.number_of_classes(), None, LabelQuerier(None, dataset.target, 200))

    # task = ClusteringTask(clusterer, "iris", None, None, LabelQuerier(None, None, 200), "cosc_test.txt")
    # clusterer.fit(dataset.data, dataset.number_of_classes(), None, LabelQuerier(None, dataset.target, 200))
    # task.run(dataset)

if __name__ == '__main__':
    # matlab_test()
    run_npu_COSC()