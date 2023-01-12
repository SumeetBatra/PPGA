import argparse
import matplotlib.pyplot as plt
import glob
import scienceplots
import csv
import pandas as pd
import os
import numpy as np

from utils.compile_cdfs import compile_cdf
from attrdict import AttrDict

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


def make_cdf_plot(cfg, data: list, ax: plt.axis):
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    y_label = "Threshold Percentage"

    # Color mapping for algorithms
    palette = {
        "CMA-MAE": "C0",
        "CMA-ME": "C1",
        "MAP-Elites (line)": "C2",
        "MAP-Elites": "C3",
    }

    x_data = [row[2] for row in data[1:]]
    y_data = [row[3] for row in data[1:]]
    ax.plot(x_data, y_data, linewidth=3.0)
    ax.set_xlim(cfg.objective_range)
    ax.set_yticks(np.arange(0, 101, 10.0))
    ax.set_xlabel("Objective")
    ax.set_ylabel(y_label)


def plot_cdf_data():
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))

    shared_params = {
        'objective_range': (0, 5000),
        'objective_resolution': 100,
        'archive_resolution': 2500,
        'skip_len': 200,
        'algorithm_name': 'cma_mae_100_0.01'
    }

    qdppo_cfg = AttrDict({
        'experiment_path': 'experiments/paper_qdppo_walker2d/1111',
        'archive_path': 'experiments/paper_qdppo_walker2d/1111/checkpoints/cp_00002000/archive_00002000.pkl',
    })
    qdppo_cfg.update(shared_params)

    pgame_cfg = AttrDict({
        'experiment_path': '/home/sumeet/QDax/experiments/pga_me_walker2d_uni_baseline_seed_1111_v2',
        'archive_path': '/home/sumeet/QDax/experiments/pga_me_walker2d_uni_baseline_seed_1111_v2/checkpoints/checkpoint_00399/pgame_archive.pkl',
    })
    pgame_cfg.update(shared_params)

    qdppo_cdf = compile_cdf(qdppo_cfg)
    pgame_cdf = compile_cdf(pgame_cfg)

    make_cdf_plot(qdppo_cfg, qdppo_cdf, axs[0])
    make_cdf_plot(pgame_cfg, pgame_cdf, axs[1])

    plt.show()


if __name__ == '__main__':
    # args = parse_args()
    # plot_qd_results(args)

    plot_cdf_data()
