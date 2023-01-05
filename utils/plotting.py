import argparse
import matplotlib.pyplot as plt
import glob
import scienceplots
import csv
import pandas as pd
import os
import numpy as np

plt.style.use('science')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', type=str)

    args = parser.parse_args()
    return vars(args)


def plot_qd_results(args):
    csv_dir = args['csv_dir']
    coverage_fp = glob.glob(csv_dir + '/' + '*coverage.csv')[0]
    coverage_data = pd.read_csv(coverage_fp).filter(regex='coverage')

    qdscore_fp = glob.glob(csv_dir + '/' + '*qdscore.csv')[0]
    qdscore_data = pd.read_csv(qdscore_fp).filter(regex='QD Score')

    bestscore_fp = glob.glob(csv_dir + '/' + '*bestscore.csv')[0]
    bestscore_data = pd.read_csv(bestscore_fp).filter(regex='best score')

    avg_perf_fp = glob.glob(csv_dir + '/' + '*avgperf.csv')[0]
    avg_perf_data = pd.read_csv(avg_perf_fp).filter(regex='average performance')

    x = np.arange(0, len(coverage_data))[:-(len(coverage_data) % 1000)]

    fig, axs = plt.subplots(1, 4, figsize=(18, 3))
    for ax in axs:
        ax.set_xlabel('Iteration')

    def plot_single_graph(ax, dataframe, title):
        minvals = dataframe.filter(regex='MIN').to_numpy().flatten()[:len(x)]
        maxvals = dataframe.filter(regex='MAX').to_numpy().flatten()[:len(x)]
        meanvals = dataframe.filter(regex='^(?!.*(MIN|MAX)).*$').to_numpy().flatten()[:len(x)]

        ax.plot(x, meanvals)
        ax.fill_between(x, minvals, maxvals, alpha=0.2)
        ax.set_title(title)

    plot_single_graph(axs[0], coverage_data, 'Coverage')
    plot_single_graph(axs[1], qdscore_data, 'QD Score')
    plot_single_graph(axs[2], bestscore_data, 'Best Fitness')
    plot_single_graph(axs[3], avg_perf_data, 'Average Fitness')
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    plot_qd_results(args)
