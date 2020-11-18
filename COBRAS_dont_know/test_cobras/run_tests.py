from multiprocessing import RLock
from multiprocessing.pool import Pool
from tqdm import tqdm


def run_test(run):
    run.run()


def run_tests_from_generator(generator, nb_of_cores=4):
    tests = generator
    if nb_of_cores > 1:
        if len(tests) == 0:
            print("Already calculated")
            return
        print("Running with {} cores".format(nb_of_cores))
        with Pool(nb_of_cores, initializer=tqdm.set_lock, initargs=(RLock(),)) as pool:
            results = pool.imap(run_test, tests)
            for _ in tqdm(results, total=len(tests)):
                pass
    else:
        print("Running sequentially on single core")
        for test in tqdm(tests):
            run_test(test)
