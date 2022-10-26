import argparse
import random
from random import shuffle
from tqdm import tqdm
import numpy as np
import time
import matplotlib.pyplot as plt


def insertion_sort(data):
    """
    Insertion sort, sort the given data by using insertion
    :param data: data to be sorted
    :return: None
    """
    for i in range(len(data)):
        j = i
        while j - 1 >= 0 and data[j] < data[j - 1]:
            data[j], data[j - 1] = data[j - 1], data[j]
            j -= 1
    return


def merge(data, left, right):
    """
    Function help merge sort to merge the left and right side to the main one
    :param data: list of data to merge left and right to
    :param left: left data to be merged
    :param right: right data to be merged
    :return: None
    """
    l = r = 0
    len_left = len(left)
    len_right = len(right)
    while l < len_left:
        if r == len_right or left[l] < right[r]:
            data[l + r] = left[l]
            l += 1
        else:
            data[l + r] = right[r]
            r += 1

    while r < len_right:
        data[l + r] = right[r]
        r += 1


def merge_sort(data):
    """
    Sorting given data by using merge sort, divide left and right mostly equal, sort them, and merge them back
    :param data: Data to be sorted
    :return:
    """
    if len(data) < 2:
        return
    left = data[:len(data) // 2]
    right = data[len(data) // 2:]

    merge_sort(left)
    merge_sort(right)
    merge(data, left, right)


def hybrid_sort(data, *, threshold=22):
    """
    Sorting given data by using hybrid sort, performing merge sort until length of data less than threshold,
    using insertion sort instead of merge sort after reaching threshold
    :param data: Data to be sorted
    :param threshold: Threshold to be used insertion
    :return:
    """
    if len(data) < 2:
        return
    if len(data) <= threshold:
        insertion_sort(data)
        return

    left = data[:len(data) // 2]
    right = data[len(data) // 2:]

    hybrid_sort(left, threshold=threshold)
    hybrid_sort(right, threshold=threshold)
    merge(data, left, right)


def measure_runtime(algorithms, sizes, trial: int):
    """
    Measure running time of the given algorithm with various input sizes, repeated several times based on trail
    :param algorithms: Algorithm to measure running time
    :param sizes: Size that input need to be generated
    :param trial: Repeated number of experiment
    :return: None
    """
    average_times = {}
    for sort_name, sorting in algorithms.items():
        thresholds = [""]
        if sort_name == "Hybrid Sort:":
            sorting, thresholds = sorting
        for threshold in thresholds:
            algo_name = sort_name + str(threshold)
            average_times[algo_name] = []
            for size in sizes:
                # Make sure generated input are the same by specific seed
                random.seed(431)
                total_time = 0
                data = list(range(size))
                for _ in tqdm(range(trial), desc=str(algo_name) + " of size " + str(size)):
                    cur_round = data[:]
                    shuffle(cur_round)
                    start = time.perf_counter()
                    sorting(cur_round) if algo_name != "Hybrid Sort:" else sorting(cur_round, threshold=threshold)
                    end = time.perf_counter()
                    total_time += end - start
                average_times[algo_name].append(total_time / trial)
    return average_times


def plot_figure(figure_name, running_time, sizes, log_scale=False, fig_name="figure"):
    """
    Plotting figure based on given running_time
    :param figure_name: figure name on the plot
    :param running_time: Dictionary of algorithm name as key and time to be plots
    :param sizes: sizes of input correspond to time
    :param log_scale: whether plot as log scale or normal scale
    :return:
    """
    plt.figure(figsize=(12, 8), dpi=100)

    for algorithm, runtime in running_time.items():

        runtime = np.array(runtime) * 1000  # Changing from sec to ms

        if log_scale:
            runtime = np.log10(runtime)

        plt.plot(sizes, runtime, label=algorithm)

    plt.legend()
    ylabel = "Log_10 of Running Time(ms)" if log_scale else "Running Time(ms)"
    plt.ylabel(ylabel)

    plt.xlabel("Input Size")

    plt.title(figure_name)

    plt.savefig(fig_name + '.png', format='png')

    plt.show()


if __name__ == "__main__":
    # User Input
    parser = argparse.ArgumentParser(description="Measure Runtime For Insertion, Merge, and Hybrid sort")
    parser.add_argument('--include_insertion', default=False, type=bool)
    parser.add_argument('--log_scale', default=False, type=bool)
    parser.add_argument('--num_trial', default=1000, type=int)
    args = parser.parse_args()

    sorting_algorithm = {"Merge Sort": merge_sort,
                         "Hybrid Sort:": (hybrid_sort, [10, 15, 20, 22, 50, 100, 500])}
    figure_name = "Merge Sort vs Hybrid Sort with several thresholds"
    sizes = [10 ** (i // 3) * (8 if i % 3 == 2 else 4 if i % 3 == 1 else 1) for i in range(13)]
    if args.include_insertion:
        # Reducing the size since it's comparing performance with insertion
        sizes = [50 * i for i in range(1, 16)]
        figure_name = "Insertion Sort vs " + figure_name
        sorting_algorithm["Insertion Sort"] = insertion_sort

    runtime = measure_runtime(sorting_algorithm, sizes, args.num_trial)
    plot_figure(figure_name, runtime, sizes, log_scale=args.log_scale)
    plot_figure(figure_name, runtime, sizes, log_scale=True, fig_name="figure_log")
