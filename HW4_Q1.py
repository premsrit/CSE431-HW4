import random
from random import shuffle
from tqdm import tqdm
import numpy as np
import time
import matplotlib.pyplot as plt


def insertion_sort(data):
    for i in range(len(data)):
        j = i
        while j - 1 >= 0 and data[j] < data[j - 1]:
            data[j], data[j - 1] = data[j - 1], data[j]
            j -= 1
    return


def merge(data, left, right):
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
    if len(data) < 2:
        return
    left = data[:len(data) // 2]
    right = data[len(data) // 2:]

    merge_sort(left)
    merge_sort(right)
    merge(data, left, right)


def measure_runtime(algorithms, sizes, trial: int):
    average_times = {}
    for sort_name, sorting in algorithms.items():
        average_times[sort_name] = []
        for size in sizes:
            random.seed(431)
            total_time = 0
            data = list(range(size))
            for _ in tqdm(range(trial), desc=str(sort_name) + " of size " + str(size)):
                cur_round = data[:]
                shuffle(cur_round)
                start = time.perf_counter()
                sorting(cur_round)
                end = time.perf_counter()
                total_time += end - start
            average_times[sort_name].append(total_time / trial)
    return average_times


def plot_figure(running_time, sizes):
    plt.figure(figsize=(12, 8), dpi=100)
    for algorithm, runtime in running_time.items():
        runtime = np.array(runtime) * 1000  # Changing from sec to ms
        plt.plot(sizes, runtime, label=algorithm)
    plt.legend()
    plt.ylabel("Running Time(ms)")
    plt.xlabel("Input Size")
    plt.title("Insertion Sort VS Merge Sort")
    plt.savefig('figure.png', format='png')
    plt.show()


if __name__ == "__main__":
    sorting_algorithm = {"Insertion sort": insertion_sort, "Merge sort": merge_sort}
    sizes = [5 * i for i in range(1, 21)]
    runtime = measure_runtime(sorting_algorithm, sizes, 1000)
    plot_figure(runtime, sizes)
