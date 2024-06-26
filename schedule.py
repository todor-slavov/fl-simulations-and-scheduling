import math
import numpy as np
import time

from logging import DEBUG, INFO, WARNING
from flwr.common.logger import log

from attempt import shuffle_locally

def read_schedule(file_path):
    with open(file_path, 'r') as file:
        # Read the entire file content
        content = file.read()
    
    # Split the content by commas and convert to integers
    array = [int(x.strip()) for x in content.split(',')]
    
    seed = 123

    np.random.seed(seed)
    array = np.random.permutation(array)

    log(INFO, f"Schedule seed: {seed}")
    llog(INFO, array[:150])

    return array[:150]

def get_non_IID_schedule(length):
    # time_seed = int(time.time())
    time_seed = 1718809444
    # set_seed = 12345678
    # np.random.seed(int(time_seed))
    np.random.seed(time_seed)
    log(INFO, f"Schedule distribution {time_seed}")

    # distribution = [10, 10, 5, 5, 3, 3, 3, 2, 2, 1]
    distribution = [1 for _ in range(10)]
    distribution = np.random.permutation(distribution)

    log(INFO, distribution)

    schedule = []

    for index, count in enumerate(distribution):
        schedule.extend([index] * count)

    schedule = schedule * (math.ceil(length / len(schedule)))

    schedule = np.random.permutation(schedule)
    log(INFO, schedule)

    return schedule


def get_IID_schedule(length):
    schedule = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * math.ceil(length / 10)

    schedule = shuffle_locally(schedule, 4)

    return schedule