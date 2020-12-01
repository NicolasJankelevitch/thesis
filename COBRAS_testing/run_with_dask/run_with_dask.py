import math
import traceback

import distributed
from tqdm import tqdm
from distributed import Client, Future
from collections import deque
from datasets import Dataset
from generate_clusterings.clustering_task import ClusteringTask

SCHEDULER_HOSTNAME = 'pinac38.cs.kuleuven.be:8786'
def scatter_datasets(client):
    """
        Scatter the preprocessed datasets over all the workers

    """
    dataset_dict = {}
    for dataset in Dataset.datasets(preprocessed=True):
        dataset_dict[dataset.name] = dataset
    dataset_dict_futures = client.scatter(dataset_dict, broadcast=True)
    return dataset_dict_futures

def chunks(l, chunk_size):
    for i in range(0, len(l), chunk_size):
        yield l[i:i + chunk_size]

def fill_dataset_in_clustering_tasks(clustering_tasks, dataset_dict):
    for clustering_task in clustering_tasks:
        clustering_task.set_dataset(dataset_dict[clustering_task.dataset_name])
    return clustering_tasks

def fill_dataset_requirements(tuple_list, client, dataset_dict):
    futures = []
    for dataset_name, func_future in tuple_list:
        dataset_future = dataset_dict[dataset_name]
        future = func_future(client, dataset_future)
        futures.append(future)
    return futures

def delayed_dataset_submission_function(experiment_function, dataset_name, *args):
    """
        A more specific functools.partial
    """
    def inner_function(client:Client, dataset_future):
        return client.submit(experiment_function, dataset_future, *args, pure=False)
    return (dataset_name, inner_function)

def execute_list_of_clustering_tasks(clustering_tasks, tests_per_batch=None):
    if len(clustering_tasks) == 0:
        print("all clustering tasks done")
    if tests_per_batch is None:
        tests_per_batch = len(clustering_tasks)

    with Client(address=SCHEDULER_HOSTNAME) as client:
        with tqdm(total=len(clustering_tasks)) as pbar:

            tasks_to_still_do = deque(clustering_tasks)
            dataset_dict = scatter_datasets(client)
            futures = []
            for _ in range(min(tests_per_batch,len(clustering_tasks))):
                task = tasks_to_still_do.popleft()
                futures.append(client.submit(ClusteringTask.run, task, dataset_dict[task.dataset_name], pure=False))

            as_completed_futures = distributed.as_completed(futures, with_results=True, raise_errors=False)
            for future, _ in as_completed_futures:
                pbar.update(1)
                f : Future = future
                if f.exception() is not None:
                    t = f.traceback()
                    # for line in t.format():
                    #     print(line, end = "")
                    # traceback.print_exception()f.exception()
                    traceback.print_tb(f.traceback())
                    print(f.exception())
                if len(tasks_to_still_do)>0:
                    task = tasks_to_still_do.popleft()
                    future = client.submit(ClusteringTask.run, task, dataset_dict[task.dataset_name], pure=False)
                    as_completed_futures.add(future)

def execute_list_of_clustering_tasks_chunked(clustering_tasks, tests_per_batch = None):
    if tests_per_batch is None:
        tests_per_batch = len(clustering_tasks)

    with Client(address=SCHEDULER_HOSTNAME) as client:
        nb_of_chunks = math.ceil(len(clustering_tasks)/tests_per_batch)
        for chunk in tqdm(chunks(clustering_tasks, tests_per_batch), desc="chunks", total= nb_of_chunks):
            dataset_dict = scatter_datasets(client)
            # tasks_ready_to_execute = fill_dataset_in_clustering_tasks(chunk, dataset_dict)
            futures = []
            for task in chunk:
                futures.append(client.submit(ClusteringTask.run, task, dataset_dict[task.dataset_name], pure=False))
            # futures = client.map(lambda x: ClusteringTask.run(x), tasks_ready_to_execute, pure = False)
            for _ in tqdm(distributed.as_completed(futures, with_results=True, raise_errors=False), total=len(futures), desc="tasks in chunk", leave=False):
                pass
            # client.restart()

def execute_list_of_futures_with_dataset_requirements(tuple_list, tests_per_batch = None):
    """

    :param tuple_list: list tuples (dataset_name, f(dataset_future) -> future
    :param tests_per_batch: how much tests per batch you want
    :return:
    """
    if tests_per_batch is None:
        tests_per_batch = len(tuple_list)

    with Client(address=SCHEDULER_HOSTNAME) as client:
        for chunk in chunks(tuple_list, tests_per_batch):
                dataset_dict = scatter_datasets(client)
                futures = fill_dataset_requirements(chunk, client, dataset_dict)
                for _ in tqdm(distributed.as_completed(futures, with_results=True, raise_errors=True), total = len(futures)):
                    pass
                client.restart()





