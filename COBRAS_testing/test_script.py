from time import sleep

from tqdm import tqdm


def test_tqdm():
    for i in tqdm(list(range(0,100)), desc="outer loop"):
        for j in tqdm(list(range(0,1000)), desc="inner loop", leave = False):
            sleep(0.01)
# test_MPCK_means_java()
# print_test_example_oracle()
test_tqdm()