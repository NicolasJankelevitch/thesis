import os
import numpy as np
import matplotlib.pyplot as plt
from util.config import DATASET_PATH

COBRAS_PAPER_DIR = "cobras-paper"
ORIGINAL_DIR = "non_preprocessed"
SIMPLE_DATA_DIR = "simple-data"


def print_dataset_info(preprocessed):
    for dataset in Dataset.datasets(preprocessed=preprocessed):
        print(dataset.name, dataset.data.shape, "nb of classes", dataset.number_of_classes())


class Dataset():
    @staticmethod
    def get_coprf_dataset_names():
        return ["ionosphere", "iris", "segmentation", "parkinsons", "glass"]
    @staticmethod
    def get_dataset_names():
        return [name[0:-5] for name in os.listdir(os.path.join(DATASET_PATH, COBRAS_PAPER_DIR))]

    @staticmethod
    def interesting_2d_datasets():
        return ['aggregation', 'compound', 'flame', 'jain', 'pathbased', 'spiral']

    @staticmethod
    def get_standard_dataset_names():
        return ["breast-cancer-wisconsin",
                "column_2C",
                "dermatology",
                "ecoli",
                "faces_expression_imagenet",
                "faces_eyes_imagenet",
                "faces_identity_imagenet",
                "faces_pose_imagenet",
                "glass",
                "hepatitis",
                "ionosphere",
                "iris",
                "newsgroups_diff3",
                "newsgroups_sim3",
                "optdigits389_full",
                "parkinsons",
                "segmentation",
                "sonar",
                "spambase",
                "wine",
                "yeast"]

    @staticmethod
    def get_quick_dataset_names():
        return ["dermatology",
                "ecoli",
                "glass",
                "ionosphere",
                "iris",
                "segmentation",
                "sonar",
                "spambase",
                "wine",
                "yeast"]
    @classmethod
    def get_non_face_news_datasets(cls):
        return [Dataset(name) for name in cls.get_non_face_news_spam_names()]

    @classmethod
    def get_non_face_news_spam_names(cls):
        allnames = cls.get_standard_dataset_names()
        return [name for name in allnames if "face" not in name and "news" not in name and "spambase" not in name]

    @classmethod
    def datasets(cls, preprocessed = False):
        if preprocessed:
            return map(lambda x: Dataset(x), cls.get_dataset_names())
        else:
            return [Dataset(x, preprocessed) for x in cls.get_coprf_dataset_names()]

    def __init__(self, name: str):
        if name.endswith(".data"):
            name = name[:-5]
        elif name.endswith(".txt"):
            name = name[:-4]
        self.name = name

        if os.path.isfile(os.path.join(DATASET_PATH,COBRAS_PAPER_DIR, name+".data")):
            self.filename = name +".data"
            data = np.loadtxt(os.path.join(DATASET_PATH, COBRAS_PAPER_DIR, self.filename), delimiter=',')
            self.data = data[:, 1:]
            self.target = data[:, 0]
        elif os.path.isfile(os.path.join(DATASET_PATH, SIMPLE_DATA_DIR, name+".txt")):
            self.filename = name +".txt"
            data = np.loadtxt(os.path.join(DATASET_PATH, SIMPLE_DATA_DIR, name + '.txt'), delimiter='\t')
            self.data = data[:, :2]
            self.target = data[:, 2]
        else:
            print(os.path.abspath(os.path.join(DATASET_PATH,SIMPLE_DATA_DIR)))
            raise Exception("unknown dataset {}".format(name))

    def plot_dataset(self, save = False):
        if self.data.shape[1] != 2:
            raise Exception("plot dataset is only supported for datasets with dimension 2. This dataset {} has dimensionality {}".format(self.name, self.data.shape[1]))
        figure = plt.figure()
        cluster_ids = set(self.target)
        for cluster_id in cluster_ids:
            cluster_indices = [idx for idx,cluster in enumerate(self.target) if cluster == cluster_id]
            cluster_points = self.data[cluster_indices]
            plt.scatter(cluster_points[:,0], cluster_points[:,1])
        plt.title(self.name)

        if save:
            plt.savefig(self.name+".png")
        else:
            plt.show()
        # figure.close()
        plt.close(figure)

    def number_of_instances(self):
        return self.target.shape[0]

    def number_of_classes(self):
        return np.unique(self.target).shape[0]

    def plot_dataset_predicted(self, predicted_constraints, save = False):
        if self.data.shape[1] != 2:
            raise Exception("plot dataset is only supported for datasets with dimension 2. This dataset {} has dimensionality {}".format(self.name, self.data.shape[1]))
        figure = plt.figure()
        cluster_ids = set(self.target)
        for cluster_id in cluster_ids:
            cluster_indices = [idx for idx,cluster in enumerate(self.target) if cluster == cluster_id]
            cluster_points = self.data[cluster_indices]
            plt.scatter(cluster_points[:,0], cluster_points[:,1])
        plt.title(self.name)

        i = 0
        for predicted, helper, _ in predicted_constraints:
            A = self.data[predicted.i1, :]
            B = self.data[predicted.i2, :]
            C = self.data[helper.i1, :]
            D = self.data[helper.i2, :]

            if i<1:
                i += 0
                # Plot of original one
                xs = [C[0], D[0]]
                ys = [C[1], D[1]]
                if helper.is_ML():
                    plt.plot(xs,ys,'g')
                if helper.is_CL():
                    plt.plot(xs,ys,'r')

               # Plot of predicted one
                xs = [A[0], B[0]]
                ys = [A[1], B[1]]
                if predicted.is_ML():
                    plt.plot(xs, ys, 'g--')
                if predicted.is_CL():
                    plt.plot(xs, ys, 'r--')

        if save:
            plt.savefig(self.name+".png")
        else:
            plt.show()
        # figure.close()
        plt.close(figure)


class SimpleDataset():
    def __init__(self, name):
        data = np.loadtxt(os.path.join(DATASET_PATH, SIMPLE_DATA_DIR, name + '.txt'), delimiter='\t')
        self.data = data[:, :2]
        self.target = data[:, 2]
        self.name = name


def get_number_for_key(lookup, name):
    if name in lookup:
        return lookup.index(name) + 1
    else:
        lookup.append(name)
        return len(lookup)


def convert_string_array(array):
    lookup = []
    for i in range(len(array)):
        element = array[i]
        number = get_number_for_key(lookup, element)
        array[i] = number


def convert_data_set(dataset_name):
    # lookup = []
    filename = dataset_name +"_original.data"
    data = np.loadtxt(os.path.join(DATASET_PATH, ORIGINAL_DIR, filename), delimiter = ',')
    y = data[:,-1]
    # convert_string_array(y)
    X = data[:,:-1]
    new_data = np.zeros((y.shape[0], 1+X.shape[1]))
    new_data[:,0] = y
    new_data[:,1:] = X
    np.savetxt(os.path.join(DATASET_PATH, ORIGINAL_DIR, dataset_name+".data"), new_data, delimiter=',')


def print_all_dataset_info():
    print("preprocessed False")
    print_dataset_info(False)
    print()
    print()
    print("preprocessed True")
    print_dataset_info(True)


def show_simple_datasets():
    for dataset_name in Dataset.interesting_2d_datasets():
        dataset = Dataset(dataset_name)
        dataset.plot_dataset(save=True)


def print_sorted_datasets():
    datasets = Dataset.get_standard_dataset_names()
    sort  = sorted(datasets, key= lambda x: Dataset(x).number_of_instances())
    print(sort)


if __name__ == '__main__':
    show_simple_datasets()