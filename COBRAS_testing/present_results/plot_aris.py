import json
import matplotlib
matplotlib.use('Agg')
import numpy as np

from config import FOLD_RESULT_DIR, FIGURE_DIR
import matplotlib.pyplot as plt
import os
plt.rcParams.update({'font.size': 22})


def make_algos_dict_and_common_datasets(algo_names, dataset_names = None):
    algos = dict()
    common_dataset_names = None if dataset_names is None else set(dataset_names)
    for algo_name in algo_names:
        algo_path = os.path.join(FOLD_RESULT_DIR,algo_name,  "ari_averages.txt")

        with open(algo_path, 'r') as file:
            averages_dict = json.load(file)
        algos[algo_name] = averages_dict
        if common_dataset_names is None:
            common_dataset_names = set(averages_dict.keys())
        else:
            common_dataset_names = common_dataset_names.intersection(averages_dict.keys())
    common_dataset_names.discard("overall_average")
    return algos, common_dataset_names


def plot_average_ARI_per_dataset(comparison_name, algo_names, line_names, colors=None, dataset_names = None):
    '''
    This function will plot average ARI comparison plots for each dataset
    :param algo_names:
    :param save:
    :return:
    '''
    # read all data into algos dict and take common dataset names
    algos, common_dataset_names = make_algos_dict_and_common_datasets(algo_names, dataset_names)

    for dataset_name in common_dataset_names:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        texts = []
        for idx, (algo_name,line_name) in enumerate(zip(algo_names,line_names)):
            ari = algos[algo_name][dataset_name]
            line_color = "C{}".format(idx+1) if colors is None else colors[idx]
            plt.plot(ari, label= line_name, c=line_color)
            t=plt.text(len(ari),ari[-1],line_name,c=line_color,  horizontalalignment='left', verticalalignment='center')
            texts.append(t)
        # adjust_text(texts,only_move={'points':'y', 'text':'y'})
        title = dataset_name
        plt.title(title)
        plt.xlabel("number of queries")
        plt.ylabel("average ARI")
        # plt.ylim(0, 1)
        # plt.legend()

        path = os.path.join(FIGURE_DIR,comparison_name,"ARI per dataset", title+".png")
        os.makedirs(os.path.dirname(path), exist_ok= True)
        plt.savefig(path, dpi=300, bbox_inches = 'tight')
        plt.close(fig)

def plot_overall_average_ARI(comparison_name, algo_names, line_names, dataset_names = None, colors = None, limitsize = None, offset_dict = None):
    #read all data into algos dict and take common dataset names
    algos, common_dataset_names = make_algos_dict_and_common_datasets(algo_names,dataset_names)

    # fig = plt.figure(figsize=(10, 10))

    # plt.iof()
    fig, ax = plt.subplots(figsize = (6,4))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0.2, 1.0)
    texts = []
    for idx,(algo_name,line_name) in enumerate(zip(algo_names,line_names)):
        average = None
        for dataset_name in common_dataset_names:
            if average is None:
                if limitsize is not None:
                    average = np.array(algos[algo_name][dataset_name][:limitsize])
                else:
                    average = np.array(algos[algo_name][dataset_name])
            else:
                if limitsize is not None:
                    average = average + np.array(algos[algo_name][dataset_name][:limitsize])
                else:
                    average = average + np.array(algos[algo_name][dataset_name])
        average = average / len(common_dataset_names)
        ofset = 0 if offset_dict is None or line_name not in offset_dict else offset_dict[line_name]
        line_color = "C{}".format(idx + 1) if colors is None else colors[idx]
        t=plt.text(len(average), average[-1]+ofset, line_name, c=line_color, horizontalalignment='left', verticalalignment='center')
        plt.plot(average, label = line_name, color = line_color, linewidth = 2 )
        texts.append(t)
    # adjust_text(texts)


    # title = "Average ARI over all datasets"
    # plt.title(title)
    plt.xlabel("number of queries")
    plt.ylabel("average ARI")
    # plt.ylim(0, 1)
    # plt.legend()
    output_file_name =os.path.join(FIGURE_DIR, comparison_name, "average_ARI.png")
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
    plt.savefig(output_file_name, dpi = 300, bbox_inches ='tight')
    plt.close(fig)
