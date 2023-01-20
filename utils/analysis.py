import argparse
import json
import pickle

import matplotlib.pyplot as plt
import glob
import scienceplots
import csv
import pandas as pd
import os
import numpy as np
import copy

from collections import OrderedDict
from utils.archive_utils import pgame_checkpoint_to_objective_df, pgame_repertoire_to_pyribs_archive, reevaluate_pgame_archive, reevaluate_ppga_archive, save_heatmap
from attrdict import AttrDict
from ribs.visualize import grid_archive_heatmap

plt.style.use('science')

shared_params = OrderedDict({
    'walker2d':
        {
            'objective_range': (0, 5000),
            'objective_resolution': 100,
            'archive_resolution': 2500,
            'skip_len': 200,
            'algorithm_name': 'cma_mae_100_0.01',
            'env_cfg': {
                'env_name': 'walker2d',
                'num_dims': 2,
                'episode_length': 1000,
                'grid_size': 50
            }
        },
    'halfcheetah':
        {
            'objective_range': (0, 9000),
            'objective_resolution': 100,
            'archive_resolution': 2500,
            'skip_len': 200,
            'algorithm_name': 'cma_mae_100_0.01',
            'env_cfg': {
                'env_name': 'halfcheetah',
                'num_dims': 2,
                'episode_length': 1000,
                'grid_size': 50
            }
        },
    'humanoid':
        {
            'objective_range': (0, 10000),
            'objective_resolution': 100,
            'archive_resolution': 2500,
            'skip_len': 200,
            'algorithm_name': 'cma_mae_100_0.01',
            'env_cfg': {
                'env_name': 'humanoid',
                'num_dims': 2,
                'episode_length': 1000,
                'grid_size': 50
            }
        },
    'ant':
        {
            'objective_range': (0, 7000),
            'objective_resolution': 100,
            'archive_resolution': 10000,
            'skip_len': 200,
            'algorithm_name': 'cma_mae_100_0.01',
            'env_cfg': {
                'env_name': 'ant',
                'num_dims': 4,
                'episode_length': 1000,
                'grid_size': 10,
            }
        }
})

PGAME_DIRS = AttrDict({
    'walker2d': f'/home/sumeet/QDax/experiments/pga_me_walker2d_uni_baseline/',
    'halfcheetah': f'/home/sumeet/QDax/experiments/pga_me_halfcheetah_uni_baseline/',
    'humanoid': f'/home/sumeet/QDax/experiments/pga_me_humanoid_uni_baseline/',
    # 'ant': '/home/sumeet/QDax/experiments/pga_me_ant_uni_baseline/'
})

PPGA_DIRS = AttrDict({
    'walker2d': './experiments/paper_qdppo_walker2d',
    'halfcheetah': './experiments/paper_qdppo_halfcheetah',
    'humanoid': './experiments/paper_qdppo_humanoid'
})


def index_of(env_name):
    return list(shared_params.keys()).index(env_name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', type=str)

    args = parser.parse_args()
    return vars(args)


def compile_cdf(cfg, dataframes=None):
    num_cells = cfg.archive_resolution

    if not dataframes:
        df_dir = cfg.archive_dir
        filenames = next(os.walk(df_dir), (None, None, []))[2]  # [] if no file
        dataframes = []
        for f in filenames:
            full_path = os.path.join(df_dir, f)
            df = pd.read_pickle(full_path)
            dataframes.append(df)

    x = np.linspace(cfg.objective_range[0], cfg.objective_range[1], cfg.objective_resolution)
    all_y_vals = []
    for df in dataframes:
        y_vals = []
        df_cells = np.array(sorted(df['objective']))
        for x_val in x:
            count = len(df_cells[df_cells > x_val])
            percentage = (count / num_cells) * 100.0
            y_vals.append(percentage)
        all_y_vals.append(np.array(y_vals))

    all_y_vals = np.vstack(all_y_vals)
    mean, stddev = np.mean(all_y_vals, axis=0), np.std(all_y_vals, axis=0)

    all_data = np.vstack((x, mean, mean - stddev, mean + stddev))
    cdf = pd.DataFrame(all_data.T, columns=['Objective',
                                            'Threshold Percentage (Mean)',
                                            'Threshold Percentage (Min)',
                                            'Threshold Percentage (Max)'])

    return cdf


def plot_all_results():
    data = [
        ('PPGA', 'data/ppga_humanoid', 'humanoid'),
        ('PGA-ME', 'data/pgame_humanoid', 'humanoid'),
        ('SEP-CMA-MAE', 'data/sep_cma_mae_humanoid', 'humanoid'),
        ('PPGA', 'data/ppga_halfcheetah', 'halfcheetah'),
        ('PGA-ME', 'data/pgame_halfcheetah', 'halfcheetah'),
        ('SEP-CMA-MAE', 'data/sep_cma_mae_halfcheetah', 'halfcheetah'),
        ('PPGA', 'data/ppga_walker2d', 'walker2d'),
        ('PGA-ME', 'data/pgame_walker2d', 'walker2d'),
        ('SEP-CMA-MAE', 'data/sep_cma_mae_walker2d', 'walker2d'),
    ]

    fig, axs = plt.subplots(4, 4, figsize=(12, 8))

    for i, (alg, csv_dir, env_name) in enumerate(data):
        args = {
            'algorithm': alg,
            'csv_dir': csv_dir,
            'axes': axs,
            'env_name': env_name,
        }
        plot_qd_results(args)

    plot_cdf_data(reevaluated_archives=False, axs=axs)

    axs[0][0].set_ylabel('Coverage', size='large')
    axs[1][0].set_ylabel('QD Score', size='large')
    axs[2][0].set_ylabel('Best Reward', size='large')
    axs[3][0].set_ylabel('Archive CDF', size='large')
    fig.tight_layout()
    plt.show()


def plot_qd_results(args):
    csv_dir = args['csv_dir']
    algorithm = args['algorithm']
    env_name = args['env_name']
    x_axis_mult = 1.0
    if 'PGA-ME' in algorithm:
        x_axis_mult = 10.0  # b/c the log freq of pga-me was every 10 qd iters


    coverage_fp = glob.glob(csv_dir + '/' + '*coverage.csv')[0]
    coverage_data = pd.read_csv(coverage_fp).filter(regex='coverage')

    qdscore_fp = glob.glob(csv_dir + '/' + '*qdscore.csv')[0]
    qdscore_data = pd.read_csv(qdscore_fp).filter(regex='QD Score|qd_score')

    bestscore_fp = glob.glob(csv_dir + '/' + '*bestscore.csv')[0]
    bestscore_data = pd.read_csv(bestscore_fp).filter(regex='best score|max_fitness')

    avg_perf_fp = glob.glob(csv_dir + '/' + '*avgperf.csv')[0]
    avg_perf_data = pd.read_csv(avg_perf_fp).filter(regex='average performance|avg_fitness')

    # x = np.arange(0, len(coverage_data))[:-(len(coverage_data) % 1000)]
    x = np.arange(0, len(coverage_data)) * x_axis_mult

    axs = args['axes']
    # for ax in axs:
    #     ax.set_xlabel('Iteration')

    def plot_single_graph(ax, dataframe, label, title=None):
        minvals = dataframe.filter(regex='MIN').to_numpy().flatten()[:len(x)]
        maxvals = dataframe.filter(regex='MAX').to_numpy().flatten()[:len(x)]
        meanvals = dataframe.filter(regex='^(?!.*(MIN|MAX)).*$').to_numpy().flatten()[:len(x)]

        ax.plot(x, meanvals, label=label)
        ax.fill_between(x, minvals, maxvals, alpha=0.2)
        if title is not None:
            ax.set_title(title)

    label = args['algorithm']
    plot_single_graph(axs[0][index_of(env_name)], coverage_data, label, title=env_name.capitalize())
    plot_single_graph(axs[1][index_of(env_name)], qdscore_data, label)
    plot_single_graph(axs[2][index_of(env_name)], bestscore_data, label)
    # plot_single_graph(axs[1][1], avg_perf_data, 'Average Fitness', label)
    axs[0][0].legend()


def make_cdf_plot(cfg, data: pd.DataFrame, ax: plt.axis, standalone: bool = False):
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    y_label = "Archive CDF"

    # Color mapping for algorithms
    palette = {
        "CMA-MAE": "C0",
        "CMA-ME": "C1",
        "MAP-Elites (line)": "C2",
        "MAP-Elites": "C3",
    }

    x = data['Objective'].to_numpy().flatten()
    y_avg = data.filter(regex='Mean').to_numpy().flatten()
    y_min = data.filter(regex='Min').to_numpy().flatten()
    y_max = data.filter(regex='Max').to_numpy().flatten()
    ax.plot(x, y_avg, linewidth=1.0, label=cfg.algorithm)
    ax.fill_between(x, y_min, y_max, alpha=0.2)
    ax.set_xlim(cfg.objective_range)
    ax.set_yticks(np.arange(0, 101, 25.0))
    ax.set_xlabel("Objective")
    if standalone:
        ax.set_ylabel(y_label)
        ax.set_title(cfg.title)
        ax.legend()


def get_pgame_df(exp_dir, reevaluated_archive=False, save=False):
    out_dir = os.path.join(exp_dir, 'cdf_analysis')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    seeds = [1111, 2222, 3333, 4444]
    dataframes = []
    for seed in seeds:
        subdir = sorted(glob.glob(exp_dir + '/' + f'*{seed}*/checkpoints/checkpoint_*'))[0]
        if reevaluated_archive:
            filepath = glob.glob(subdir + '/' + '*reeval_archive*')[0]
            with open(filepath, 'rb') as f:
                df = pickle.load(f).as_pandas()
        else:
            df = pgame_checkpoint_to_objective_df(subdir)
        dataframes.append(df)
        if save:
            df.to_pickle(os.path.join(out_dir, f'scores_{seed}.pkl'))
    return dataframes


def get_qdppo_df(exp_dir, reevaluated_archive=False):
    seeds = [1111, 2222, 3333, 4444]
    dataframes = []
    for seed in seeds:
        subdir = sorted(glob.glob(exp_dir + '/' + f'*{seed}*/checkpoints/cp_*'))[-1]  # gets the most recent checkpoint
        if reevaluated_archive:
            filename = glob.glob(subdir + '/' + '*reeval_archive*')[0]
            with open(filename, 'rb') as f:
                df = pickle.load(f).as_pandas()
        else:
            filename = glob.glob(subdir + '/' + 'archive_*')[0]
            df = pd.read_pickle(filename)
        dataframes.append(df)

    return dataframes


def plot_cdf_data(reevaluated_archives=False, axs=None):
    standalone_plot = False
    if axs is None:
        standalone_plot = True
        fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    subtitle = 'Archive CDFs'
    prefix = 'Corrected ' if reevaluated_archives else ''
    title = prefix + subtitle

    for i, ((exp_name, qdppo_dir), (_, pgame_dir)) in enumerate(zip(PPGA_DIRS.items(), PGAME_DIRS.items())):
        base_cfg = AttrDict(shared_params[exp_name])
        base_cfg['title'] = exp_name

        qdppo_cfg = copy.copy(base_cfg)
        qdppo_cfg.update({'archive_dir': qdppo_dir, 'algorithm': 'QDPPO'})
        qdppo_dataframes = get_qdppo_df(qdppo_dir, reevaluated_archive=reevaluated_archives)
        qdppo_cdf = compile_cdf(qdppo_cfg, dataframes=qdppo_dataframes)

        pgame_cfg = copy.copy(base_cfg)
        pgame_cfg.update({'archive_dir': pgame_dir, 'algorithm': 'PGA-ME'})
        pgame_dataframes = get_pgame_df(pgame_dir, reevaluated_archive=reevaluated_archives)
        pgame_cdf = compile_cdf(pgame_cfg, dataframes=pgame_dataframes)

        if standalone_plot:
            (j, k) = np.unravel_index(i, (2, 2))
            make_cdf_plot(qdppo_cfg, qdppo_cdf, axs[j][k])
            make_cdf_plot(pgame_cfg, pgame_cdf, axs[j][k])
        else:
            env_idx = index_of(exp_name)
            make_cdf_plot(qdppo_cfg, qdppo_cdf, axs[3][env_idx])
            make_cdf_plot(pgame_cfg, pgame_cdf, axs[3][env_idx])



def load_and_eval_pgame_archive(exp_name, seed, data_is_saved=False):
    exp_dir = PGAME_DIRS[exp_name]
    cp_path = sorted(glob.glob(exp_dir + '/' + f'*{seed}*/checkpoints/checkpoint_*'))[0]
    save_path = cp_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    base_cfg = AttrDict(shared_params[exp_name])
    env_cfg = base_cfg.env_cfg
    env_cfg.seed = seed
    if data_is_saved:
        orig_archive_fp = glob.glob(save_path + '/' + '*original_archive*')[0]
        with open(orig_archive_fp, 'rb') as f:
            original_archive = pickle.load(f)

        new_archive_fp = os.path.join(save_path, f'{exp_name}_reeval_archive.pkl')
        with open(new_archive_fp, 'rb') as f:
            new_archive = pickle.load(f)
        print(f'{exp_name} Re-evaluated PGAME Archive \n'
              f'Coverage: {new_archive.stats.coverage} \n'
              f'Max fitness: {new_archive.stats.obj_max} \n'
              f'Avg Fitness: {new_archive.stats.obj_mean} \n'
              f'QD Score: {new_archive.offset_qd_score}')
    else:
        original_archive, pgame_sols = pgame_repertoire_to_pyribs_archive(cp_path + '/', env_cfg, save_path=save_path)
        new_archive = reevaluate_pgame_archive(env_cfg, archive_df=original_archive.as_pandas(), save_path=save_path)
    return original_archive, new_archive


def load_and_eval_ppga_archive(exp_name, seed, data_is_saved=False):
    exp_dir = PPGA_DIRS[exp_name]
    cp_path = sorted(glob.glob(exp_dir + '/' + f'*{seed}*/checkpoints/cp_*'))[-1]  # gets the most recent checkpoint
    save_path = cp_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    base_cfg = AttrDict(shared_params[exp_name])
    env_cfg = base_cfg.env_cfg
    env_cfg.seed = seed

    agent_cfg_fp = exp_dir + f'/{seed}/' + 'cfg.json'
    with open(agent_cfg_fp, 'r') as f:
        agent_cfg = json.load(f)
        agent_cfg = AttrDict(agent_cfg)

    scheduler_fp = glob.glob(cp_path + '/' + 'scheduler_*')[0]
    with open(scheduler_fp, 'rb') as f:
        scheduler = pickle.load(f)
        original_archive = scheduler.archive

    if data_is_saved:
        new_archive_fp = os.path.join(save_path, f'{exp_name}_reeval_archive.pkl')
        with open(new_archive_fp, 'rb') as f:
            new_archive = pickle.load(f)
        print(f'{exp_name} Re-evaluated PPGA Archive \n'
              f'Coverage: {new_archive.stats.coverage} \n'
              f'Max fitness: {new_archive.stats.obj_max} \n'
              f'Avg Fitness: {new_archive.stats.obj_mean} \n'
              f'QD Score: {new_archive.offset_qd_score}')
    else:
        new_archive = reevaluate_ppga_archive(env_cfg, agent_cfg=agent_cfg, original_archive=original_archive, save_path=save_path)

    return original_archive, new_archive


def visualize_reevaluated_archives():
    seed = 1111
    fig, axs = plt.subplots(2, 3, figsize=(10, 6))

    for i, exp_name in enumerate(PGAME_DIRS.keys()):
        _, new_pgame_archive = load_and_eval_pgame_archive(exp_name, seed, data_is_saved=True)
        _, new_ppga_archive = load_and_eval_ppga_archive(exp_name, seed, data_is_saved=True)

        grid_archive_heatmap(new_pgame_archive, ax=axs[0][i])
        grid_archive_heatmap(new_ppga_archive, ax=axs[1][i])
        axs[0][i].set_title(exp_name)

    axs[0][0].set_ylabel('PGA-ME')
    axs[1][0].set_ylabel('PPGA')
    fig.tight_layout()
    plt.show()


def print_corrected_qd_metrics():
    seeds = [1111, 2222, 3333, 4444]
    ppga_data = {'coverage': [],
                 'obj_max': [],
                 'obj_mean': [],
                 'qd_score': [],
                 'num_elites': [],
                 'offset_qd_score': []}
    pgame_data = copy.deepcopy(ppga_data)
    final_ppga_data, final_pgame_data = {}, {}

    for exp_name in PPGA_DIRS.keys():
        # clear any old data and start fresh
        for key in ppga_data.keys():
            ppga_data[key] = []
        for key in pgame_data.keys():
            pgame_data[key] = []
        for seed in seeds:
            _, new_ppga_archive = load_and_eval_ppga_archive(exp_name, seed, data_is_saved=True)
            _, new_pgame_archive = load_and_eval_pgame_archive(exp_name, seed, data_is_saved=True)

            for name, val in new_ppga_archive.stats._asdict().items():
                ppga_data[name].append(val)
            # this is not in stats, so we need to add it manually
            ppga_data['offset_qd_score'].append(new_ppga_archive.offset_qd_score)

            for name, val in new_pgame_archive.stats._asdict().items():
                pgame_data[name].append(val)
            pgame_data['offset_qd_score'].append(new_pgame_archive.offset_qd_score)

        # now that we've collected data from all seeds, we can average and put it into the final dict
        final_ppga_data[exp_name] = {}
        final_pgame_data[exp_name] = {}

        for name, data in ppga_data.items():
            final_ppga_data[exp_name][name] = np.mean(np.array(data))

        for name, data in pgame_data.items():
            final_pgame_data[exp_name][name] = np.mean(np.array(data))

    # once we've done this for all experiments, we can print the final result
    for exp_name, data in final_ppga_data.items():
        print(f'PPGA {exp_name}: Averaged Results: {data}')

    for exp_name, data in final_pgame_data.items():
        print(f'PGA-ME {exp_name}: Averaged Results: {data}')


if __name__ == '__main__':
    args = parse_args()
    # plot_qd_results(args)
    plot_all_results()
    # get_qdppo_df('/home/sumeet/QDPPO/experiments/paper_qdppo_walker2d')
    # plot_cdf_data(reevaluated_archives=True)
    # visualize_reevaluated_archives()
    # print_corrected_qd_metrics()
