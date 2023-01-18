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

from utils.archive_utils import pgame_checkpoint_to_objective_df, pgame_repertoire_to_pyribs_archive, reevaluate_pgame_archive, reevaluate_ppga_archive, save_heatmap
from attrdict import AttrDict
from ribs.visualize import grid_archive_heatmap

plt.style.use('science')

shared_params = {
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
            'objective_resolution': 11,
            'archive_resolution': 2500,
            'skip_len': 200,
            'algorithm_name': 'cma_mae_100_0.01',
            'env_cfg': {
                'env_name': 'humanoid',
                'num_dims': 2,
                'episode_length': 1000,
                'grid_size': 50
            }
        }
}


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


def make_cdf_plot(cfg, data: pd.DataFrame, ax: plt.axis):
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

    x = data['Objective'].to_numpy().flatten()
    y_avg = data.filter(regex='Mean').to_numpy().flatten()
    y_min = data.filter(regex='Min').to_numpy().flatten()
    y_max = data.filter(regex='Max').to_numpy().flatten()
    ax.plot(x, y_avg, linewidth=3.0, label=cfg.algorithm)
    ax.fill_between(x, y_min, y_max, alpha=0.2)
    ax.set_xlim(cfg.objective_range)
    ax.set_yticks(np.arange(0, 101, 10.0))
    ax.set_xlabel("Objective")
    ax.set_ylabel(y_label)
    ax.set_title(cfg.title)
    ax.legend()


def get_pgame_df(exp_dir, save=False):
    out_dir = os.path.join(exp_dir, 'cdf_analysis')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    seeds = [1111, 2222, 3333, 4444]
    dataframes = []
    for seed in seeds:
        subdir = sorted(glob.glob(exp_dir + '/' + f'*{seed}*/checkpoints/checkpoint_*'))[0]
        df = pgame_checkpoint_to_objective_df(subdir)
        dataframes.append(df)
        if save:
            df.to_pickle(os.path.join(out_dir, f'scores_{seed}.pkl'))
    return dataframes


def get_qdppo_df(exp_dir):
    seeds = [1111, 2222, 3333, 4444]
    dataframes = []
    for seed in seeds:
        subdir = sorted(glob.glob(exp_dir + '/' + f'*{seed}*/checkpoints/cp_*'))[-1]  # gets the most recent checkpoint
        filename = glob.glob(subdir + '/' + 'archive_*')[0]
        df = pd.read_pickle(filename)
        dataframes.append(df)

    return dataframes


def plot_cdf_data():
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))

    qdppo_dirs = AttrDict({
        'walker2d': 'experiments/paper_qdppo_walker2d',
        'halfcheetah': 'experiments/paper_qdppo_halfcheetah',
        'humanoid': 'experiments/paper_qdppo_humanoid',
    })

    pgame_dirs = AttrDict({
        'walker2d': '/home/sumeet/QDax/experiments/pga_me_walker2d_uni_baseline',
        'halfcheetah': '/home/sumeet/QDax/experiments/pga_me_halfcheetah_uni_baseline',
        'humanoid': '/home/sumeet/QDax/experiments/pga_me_humanoid_uni_baseline'
    })

    for i, ((exp_name, qdppo_dir), (_, pgame_dir)) in enumerate(zip(qdppo_dirs.items(), pgame_dirs.items())):
        base_cfg = AttrDict(shared_params[exp_name])
        base_cfg['title'] = exp_name

        qdppo_cfg = copy.copy(base_cfg)
        qdppo_cfg.update({'archive_dir': qdppo_dir, 'algorithm': 'QDPPO'})
        qdppo_dataframes = get_qdppo_df(qdppo_dir)
        qdppo_cdf = compile_cdf(qdppo_cfg, dataframes=qdppo_dataframes)

        pgame_cfg = copy.copy(base_cfg)
        pgame_cfg.update({'archive_dir': pgame_dir, 'algorithm': 'PGA-ME'})
        pgame_dataframes = get_pgame_df(pgame_dir)
        pgame_cdf = compile_cdf(pgame_cfg, dataframes=pgame_dataframes)

        (j, k) = np.unravel_index(i, (2, 2))
        make_cdf_plot(qdppo_cfg, qdppo_cdf, axs[j][k])
        make_cdf_plot(pgame_cfg, pgame_cdf, axs[j][k])

    fig.tight_layout()
    plt.show()


def visualize_reevaluated_archives():
    seed = 1111
    fig, axs = plt.subplots(2, 3, figsize=(10, 6))

    pgame_dirs = AttrDict({
        'walker2d': f'/home/sumeet/QDax/experiments/pga_me_walker2d_uni_baseline/',
        'halfcheetah': f'/home/sumeet/QDax/experiments/pga_me_halfcheetah_uni_baseline/',
        'humanoid': f'/home/sumeet/QDax/experiments/pga_me_humanoid_uni_baseline/'
    })

    ppga_dirs = AttrDict({
        'walker2d': './experiments/paper_qdppo_walker2d',
        'halfcheetah': './experiments/paper_qdppo_halfcheetah',
        'humanoid': './experiments/paper_qdppo_humanoid'
    })

    def load_and_eval_pgame_archive(exp_name, data_is_saved=False):
        exp_dir = pgame_dirs[exp_name]
        cp_path = sorted(glob.glob(exp_dir + '/' + f'*{seed}*/checkpoints/checkpoint_*'))[0]
        save_path = os.path.join(exp_dir, 'posthoc_analysis')
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        base_cfg = AttrDict(shared_params[exp_name])
        env_cfg = base_cfg.env_cfg
        env_cfg.seed = seed
        if data_is_saved:
            orig_archive_fp = os.path.join(save_path, f'{exp_name}_original_archive.pkl')
            with open(orig_archive_fp, 'rb') as f:
                original_archive = pickle.load(f)

            new_archive_fp = os.path.join(save_path, f'{exp_name}_reeval_archive.pkl')
            with open(new_archive_fp, 'rb') as f:
                new_archive = pickle.load(f)
            print(f'Re-evaluated PGAME Archive \n'
                  f'Coverage: {new_archive.stats.coverage} \n'
                  f'Max fitness: {new_archive.stats.obj_max} \n'
                  f'Avg Fitness: {new_archive.stats.obj_mean} \n'
                  f'QD Score: {new_archive.offset_qd_score}')
        else:
            original_archive, pgame_sols = pgame_repertoire_to_pyribs_archive(cp_path + '/', env_cfg, save_path=save_path)
            new_archive = reevaluate_pgame_archive(env_cfg, archive_df=original_archive.as_pandas(), save_path=save_path)
        return original_archive, new_archive

    def load_and_eval_ppga_archive(exp_name, data_is_saved=False):
        exp_dir = ppga_dirs[exp_name]
        cp_path = sorted(glob.glob(exp_dir + '/' + f'*{seed}*/checkpoints/cp_*'))[-1]  # gets the most recent checkpoint
        save_path = os.path.join(exp_dir, 'posthoc_analysis')
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
            print(f'Re-evaluated PPGA Archive \n'
                  f'Coverage: {new_archive.stats.coverage} \n'
                  f'Max fitness: {new_archive.stats.obj_max} \n'
                  f'Avg Fitness: {new_archive.stats.obj_mean} \n'
                  f'QD Score: {new_archive.offset_qd_score}')
        else:
            new_archive = reevaluate_ppga_archive(env_cfg, agent_cfg=agent_cfg, original_archive=original_archive, save_path=save_path)

        return original_archive, new_archive

    for i, exp_name in enumerate(pgame_dirs.keys()):
        _, new_pgame_archive = load_and_eval_pgame_archive(exp_name, data_is_saved=True)
        _, new_ppga_archive = load_and_eval_ppga_archive(exp_name, data_is_saved=True)

        grid_archive_heatmap(new_pgame_archive, ax=axs[0][i])
        grid_archive_heatmap(new_ppga_archive, ax=axs[1][i])
        axs[0][i].set_title(exp_name)

    axs[0][0].set_ylabel('PGA-ME')
    axs[1][0].set_ylabel('PPGA')
    fig.tight_layout()
    plt.show()




if __name__ == '__main__':
    # args = parse_args()
    # plot_qd_results(args)
    # get_qdppo_df('/home/sumeet/QDPPO/experiments/paper_qdppo_walker2d')
    # plot_cdf_data()
    visualize_reevaluated_archives()
