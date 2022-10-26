import argparse
import random
from random import shuffle
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
import time
import matplotlib.pyplot as plt
from MyRbTree import RBtree as myRBTree
from RbTree import RedBlackTree as RBTree


def generate_key():
    """
    Generate random alphabet key as the key to dictionary
    :return: str represent key
    """
    size = random.randint(0, 100)
    generate = [chr(ord('a') + random.randint(0, 25)) for _ in range(size)]
    return "".join(generate)


def measure_time(data_structures, input_sizes, trial, generate_repeated=2.0, search_only=False, delete_rate=0.2):
    """
    Measuring insert/search time for given data structure (OrderDict, dict) by given parameters
    :param delete_rate: whether input to be del key or change key if key already exist
    :param data_structures: Data structure to measure running time
    :param input_sizes: Size that input need to be generated
    :param trial: Repeated number of experiment
    :param generate_repeated: Indicated whether input need to guarantee repeated or not
    :param search_only: Indicated whether only measure only search time or not
    :return: None
    """
    average_time = {}
    for size in input_sizes:
        random.seed(431)
        all_keys = []
        check_keys = {}
        inputs = []
        generate_size = max(1, int(size * generate_repeated))
        # Generate input to insert/search
        for _ in tqdm(range(generate_size), desc="Generate Input of size " + str(generate_size)):
            key = generate_key()
            while key in check_keys:
                key = generate_key()
            check_keys[key] = 1
            all_keys.append(key)
            # Random value as value of specific key
            inputs.append((key, random.randint(-10 ** 6, 10 ** 6)))
        for structure_name, data_structure in data_structures.items():
            random.seed(431)
            total_time = 0
            if structure_name not in average_time:
                average_time[structure_name] = []
            current_structure = data_structure()
            if search_only:
                cur_input = [inputs[random.randint(0, len(inputs) - 1)] for _ in range(size)]
                # Pre-insert some values
                for key, val in cur_input:
                    current_structure[key] = val

            for _ in tqdm(range(trial), desc=str(structure_name) + " of size " + str(size)):
                cur_input = [inputs[random.randint(0, len(inputs) - 1)] for _ in range(size)]
                start = time.perf_counter()
                for key, val in cur_input:
                    # Only search and change existing key, if not continue
                    if search_only and key not in current_structure:
                        continue

                    if key in current_structure:
                        # Delete key
                        if random.random() < delete_rate:
                            del current_structure[key]
                            continue

                    current_structure[key] = val

                end = time.perf_counter()

                total_time += (end - start)

            average_time[structure_name].append(total_time / trial)
    return average_time


def plot_figure(figure_name, running_time, sizes, log_scale=False, save_name="figure"):
    """
    Plotting figure based on given running_time
    :param figure_name: figure name on the plot
    :param running_time: Dictionary of algorithm name as key and time to be plots
    :param sizes: sizes of input correspond to time
    :param log_scale: whether plot as log scale or normal scale
    :param save_name: Saving file name
    :return: None
    """
    plt.figure(figsize=(12, 8), dpi=100)

    for algorithm, runtime in running_time.items():

        runtime = np.array(runtime)

        if log_scale:
            runtime = np.log(runtime)

        plt.plot(sizes, runtime, label=algorithm)

    plt.legend()
    ylabel = "Log_10 of Running Time(s)" if log_scale else "Running Time(s)"
    plt.ylabel(ylabel)

    plt.xlabel("Input Size")

    plt.title(figure_name)

    plt.savefig(save_name + '.png', format='png')

    plt.show()


if __name__ == "__main__":
    # User Input
    parser = argparse.ArgumentParser(description="Measure Runtime For various back-end of Hashtable")
    parser.add_argument('--log_scale', default=False, type=bool)
    parser.add_argument('--num_trial', default=1, type=int)
    args = parser.parse_args()
    sizes = [10 ** (i//2) * (5 if i % 2 else 1) for i in range(12)]
    data_structure = {"dict": dict, "my red-black tree": myRBTree, "given red-black tree": RBTree}
    time1 = measure_time(data_structure, sizes, args.num_trial, generate_repeated=0.5)
    plot_figure("Dict VS Other Back-end structures", time1, sizes, log_scale=args.log_scale, save_name="figure1")

    time_search = measure_time(data_structure, sizes, args.num_trial, generate_repeated=0.5, search_only=True)
    plot_figure("Dict VS Other Back-end structures (Search Only)", time_search, sizes, log_scale=args.log_scale, save_name="figure2")