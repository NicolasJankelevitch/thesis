from multiprocessing import RLock
from multiprocessing.pool import Pool

from tqdm import tqdm

from datasets import Dataset


def run_test(run):
    run.run()

def run_clustering_task(argument_tuple):
    clustering_task, dataset = argument_tuple
    clustering_task.run(dataset)

def run_clustering_tasks_locally(clustering_tasks, nb_of_cores =3 ):
    dataset_dict = {}
    for dataset in Dataset.datasets(preprocessed=True):
        dataset_dict[dataset.name] = dataset
    tuple_list = []
    for clustering_task in clustering_tasks:
        tuple_list.append((clustering_task, dataset_dict[clustering_task.dataset_name]))
    if nb_of_cores > 1:
        with Pool(nb_of_cores, initializer=tqdm.set_lock, initargs=(RLock(),)) as pool:
            results = pool.imap(run_clustering_task, tuple_list)
            for _ in tqdm(results, total=len(tuple_list)):
                pass
    else:
        for clustering_task, dataset in tqdm(tuple_list):
            clustering_task.run(dataset)


def run_tests_from_generator(generator, nb_of_cores = 3):
    tests = generator
    if nb_of_cores > 1:
        if len(tests) == 0:
            print("already calculated")
            return
        print("running with {} cores".format(nb_of_cores))
        with Pool(nb_of_cores, initializer=tqdm.set_lock, initargs=(RLock(),)) as pool:
            results = pool.imap(run_test, tests)
            for _ in tqdm(results, total=len(tests)):
                pass
    else:
        print("running sequentially on single core")
        for test in tqdm(tests):
            run_test(test)