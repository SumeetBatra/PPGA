import copy
import argparse
import time
import os
import sys
import csv
import torch
import pickle
import numpy as np
import wandb
import shutil
from pathlib import Path

from distutils.util import strtobool
from attrdict import AttrDict
from ribs.archives import CVTArchive, GridArchive
from ribs.emitters import GradientAborescenceEmitter
from ribs.schedulers import Scheduler

from RL.ppo import PPO
from utils.utilities import log, config_wandb, get_checkpoints
from models.actor_critic import Actor
from envs.brax_custom.brax_env import make_vec_env_brax
from utils.normalize_obs import NormalizeReward
from utils.utilities import save_cfg
from utils.archive_utils import save_heatmap, load_scheduler_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    #PPO params
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, help="Choose from [QDAntBulletEnv-v0,"
                                                     "QDHalfCheetahBulletEnv-v0]")
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument("--torch_deterministic", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--use_wandb", default=False, type=lambda x: bool(strtobool(x)),
                        help='Use weights and biases to track the exp')
    parser.add_argument('--wandb_run_name', type=str, default='ppo_ant')
    parser.add_argument('--wandb_group', type=str)
    parser.add_argument('--report_interval', type=int, default=5, help='Log objective results every N updates')

    # algorithm args
    parser.add_argument('--total_timesteps', type=int, default=1000000)
    parser.add_argument('--env_type', type=str, choices=['brax', 'isaac'], help='Whether to use cpu-envs or gpu-envs for rollouts')
    # args for brax
    parser.add_argument('--env_batch_size', default=1, type=int, help='Number of parallel environments to run')

    # args for cpu-envs
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker processes to spawn. '
                             'Should always be <= number of logical cores on your machine')
    parser.add_argument('--envs_per_worker', type=int, default=1,
                        help='Num envs each worker process will step through sequentially')
    parser.add_argument('--rollout_length', type=int, default=2048,
                        help='the number of steps to run in each environment per policy rollout')
    # ppo hyperparams
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--anneal_lr', type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help='Toggle learning rate annealing for policy and value networks')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for rewards')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='Lambda discount used for general advantage est')
    parser.add_argument('--num_minibatches', type=int, default=32)
    parser.add_argument('--update_epochs', type=int, default=10, help='The K epochs to update the policy')
    parser.add_argument("--norm_adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip_coef", type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip_vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--entropy_coef", type=float, default=0.0,
                        help="coefficient of the entropy")
    parser.add_argument("--vf_coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target_kl", type=float, default=None,
                        help="the target KL divergence threshold")
    parser.add_argument('--normalize_obs', type=lambda x: bool(strtobool(x)), default=False, help='Normalize observations across a batch using running mean and stddev')
    parser.add_argument('--normalize_rewards', type=lambda x: bool(strtobool(x)), default=False, help='Normalize rewards across a batch using running mean and stddev')
    parser.add_argument('--weight_decay', type=float, default=None, help='Apply L2 weight regularization to the NNs')

    # QD Params
    parser.add_argument('--algorithm', type=str, choices=['ppo', 'qd-ppo'])
    parser.add_argument("--num_emitters", type=int, default=1, help="Number of parallel"
                                                                    " CMA-ES instances exploring the archive")
    parser.add_argument("--num_dims", type=int, help="Dimensionality of measures")
    parser.add_argument("--mega_lambda", type=int, required=True, help="Branching factor for each step of MEGA i.e. the number of branching solutions from the current solution point")
    parser.add_argument('--dqd_lr', type=float, default=0.001, help='Learning rate on gradient arborescence in DQD. Used in cma-mega, cma-maega, etc')
    parser.add_argument('--log_arch_freq', type=int, default=10, help='Frequency in num iterations at which we checkpoint the archive')
    parser.add_argument('--load_scheduler_from_cp', type=str, default=None, help='Load an existing QD scheduler from a checkpoint path')
    parser.add_argument('--pretrain', type=lambda x: bool(strtobool(x)), default=False, help='Pretrain a policy with PPO as the initial solution point for DQD')
    parser.add_argument('--total_iterations', type=int, default=100, help='Number of iterations to run the entire dqd-rl loop')
    parser.add_argument('--dqd_algorithm', type=str, choices=['cma_mega_adam', 'cma_maega'], help='Which DQD algorithm should be running in the outer loop')
    parser.add_argument('--logdir', type=str, help='Experiment results directory')
    parser.add_argument('--save_heatmaps', type=lambda x: bool(strtobool(x)), default=True, help='Save the archive heatmaps. Only applies to archives with <= 2 measures')

    args = parser.parse_args()
    cfg = AttrDict(vars(args))
    return cfg


def create_scheduler(cfg,
                     algorithm,
                     seed,
                     num_emitters,
                     learning_rate=None,
                     use_result_archive=True,
                     initial_sol=None):
    """Creates a scheduler based on the algorithm name.

    Args:
        algorithm (str): Name of the algorithm passed into sphere_main.
        solution_dim(int): Dimensionality of the sphere function.
        archive_dims (int): Dimensionality of the archive.
        learning_rate (float): Learning rate of archive.
        use_result_archive (bool): Whether to use a separate archive to store
            the results.
        seed (int): Main seed or the various components.
    Returns:
        ribs.schedulers.Scheduler: A ribs scheduler for running the algorithm.
    """
    obs_shape, action_shape = cfg.obs_shape, cfg.action_shape
    action_dim, obs_dim = np.prod(action_shape), np.prod(obs_shape)
    log.debug(f'Environment {cfg.env_name}, {action_dim=}, {obs_dim=}')
    batch_size = cfg.mega_lambda

    if initial_sol is None:
        initial_agent = Actor(cfg, obs_shape, action_shape)
        initial_sol = initial_agent.serialize()
    solution_dim = len(initial_sol)
    mode = 'batch'
    threshold_min = -np.inf

    # TODO: specify bounds and archive_dims somewhere based on what env was passed in
    bounds = [(0.0, 1.0)] * cfg.num_dims
    archive_dims = [100, 100]

    if algorithm in ["cma_mae", "cma_maega"]:
        threshold_min = 0.0

    if learning_rate is None:
        if algorithm in ["cma_mae", "cma_maega"]:
            learning_rate = 0.5
        else:
            learning_rate = 1.0

    archive = GridArchive(solution_dim=solution_dim,
                          dims=archive_dims,
                          ranges=bounds,
                          learning_rate=learning_rate,  # lr used only in maega
                          threshold_min=threshold_min,
                          seed=seed)
    result_archive = None
    if use_result_archive:
        result_archive = GridArchive(solution_dim=solution_dim,
                                     dims=archive_dims,
                                     ranges=bounds,
                                     seed=seed)

    surrogate_archive = None
    use_surrogate_archive = True
    if use_surrogate_archive:
        surrogate_archive = GridArchive(solution_dim=1,
                                        dims=[20, 20, 20, 20],
                                        ranges=bounds,
                                        learning_rate=learning_rate,
                                        threshold_min=threshold_min,
                                        seed=seed)

    # Create emitters. Each emitter needs a different seed, so that they do not
    # all do the same thing.
    emitter_seeds = [None] * num_emitters if seed is None else np.arange(
        seed, seed + num_emitters)

    if algorithm == 'cma_mega_adam':
        # Note that only one emitter is used for cma_mega_adam. This is to be
        # consistent with Fontaine 2021 <https://arxiv.org/abs/2106.03894>.
        emitters = [
            GradientAborescenceEmitter(
                archive,
                initial_sol,
                sigma0=1.0,
                step_size=cfg.dqd_lr,
                grad_opt="adam",
                selection_rule="mu",
                batch_size=batch_size,  # 1 solution is returned by ask_dqd
                seed=emitter_seeds[0],
                use_wandb=cfg.use_wandb)
        ]
    else:
        # cma-maega
        emitters = [
            GradientAborescenceEmitter(archive,
                                       initial_sol,
                                       sigma0=5.0,
                                       step_size=cfg.dqd_lr,
                                       ranker="imp",
                                       selection_rule="mu",
                                       grad_opt="adam",
                                       restart_rule="no_improvement",
                                       bounds=None,
                                       batch_size=batch_size,
                                       seed=s,
                                       use_wandb=cfg.use_wandb) for s in emitter_seeds
        ]

    log.debug(
        f"Created Scheduler for {algorithm} with a dqd learning rate {cfg.dqd_lr}, archive learning rate {learning_rate}, "
        f"and add mode {mode}, using solution dim {solution_dim} and archive "
        f"dims {archive_dims}. Min threshold is {threshold_min}")

    return Scheduler(archive, emitters, result_archive, surrogate_archive, add_mode=mode)


def run_experiment(cfg,
                   algorithm,
                   ppo,
                   itrs=10000,
                   outdir="./logs",
                   log_freq=1,
                   log_arch_freq=1000,
                   seed=None,
                   use_wandb=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Create a directory for this specific trial.
    s_logdir = os.path.join(outdir, f"{algorithm}", f"trial_0")
    logdir = Path(s_logdir)
    if not logdir.is_dir():
        logdir.mkdir()
    save_cfg(str(logdir), cfg)

    cp_dir = os.path.join(logdir, 'checkpoints')
    cp_dir = Path(cp_dir)
    if not cp_dir.is_dir():
        cp_dir.mkdir()

    heatmap_dir = os.path.join(logdir, 'heatmaps')
    heatmap_dir = Path(heatmap_dir)
    if not heatmap_dir.is_dir():
        heatmap_dir.mkdir()

    # create a new summary file
    summary_filename = os.path.join(s_logdir, f'summary.csv')
    if os.path.exists(summary_filename):
        os.remove(summary_filename)
    with open(summary_filename, 'w') as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(['Iteration', 'QD-Score', 'Coverage', 'Maximum', 'Average'])

    # cma_mega - specific params
    is_init_pop = False
    is_dqd = True
    use_result_archive = algorithm in ["cma_mae", "cma_maega"]

    # pretrain the policy to bootstrap dqd algorithm
    if cfg.pretrain:
        ppo.vec_inference.rew_normalizers[0] = NormalizeReward(cfg.num_envs)
        avg_reward = -1000.0
        while (avg_reward < 0):
            log.info(f'pretraining an initial policy. {avg_reward=}')
            ppo.train(num_updates=10, rollout_length=16, calculate_dqd_gradients=False)
            avg_reward = sum(ppo.episodic_returns) / len(ppo.episodic_returns)

    if cfg.load_scheduler_from_cp:
        log.info("Loading an existing scheduler!")
        scheduler = load_scheduler_from_checkpoint(cfg.load_scheduler_from_cp)
    else:
        scheduler = create_scheduler(cfg,
                                     algorithm,
                                     seed,
                                     cfg.num_emitters,
                                     learning_rate=None,
                                     use_result_archive=use_result_archive,
                                     initial_sol=ppo.agents[0].serialize())
    result_archive = scheduler.result_archive
    best = 0.0

    obs_shape = ppo.vec_env.single_observation_space.shape
    action_shape = ppo.vec_env.single_action_space.shape

    # save the initial heatmap
    if cfg.num_dims <= 2:
        save_heatmap(result_archive, os.path.join(str(heatmap_dir), f'heatmap_{0:05d}.png'))

    prev_mean_grad_coeffs = scheduler.emitters[0].opt.mu

    for itr in range(1, itrs + 1):
        itr_start = time.time()

        # returns a single sol per emitter
        solution_batch = scheduler.ask_dqd()
        mean_agents = [Actor(cfg, obs_shape, action_shape).deserialize(sol).to(device) for sol
                       in solution_batch]

        # # need to reset the std-dev parameter. Otherwise the agents stop learning and branched solutions
        # # all fall in the same cell
        for agent in mean_agents:
            agent.actor_logstd = torch.nn.Parameter(torch.zeros(1, np.prod(cfg.action_shape)))

        ppo.agents = mean_agents
        objs, measures, jacobian, metadata = ppo.train(num_updates=10,
                                                       rollout_length=cfg.rollout_length,
                                                       calculate_dqd_gradients=True,
                                                       negative_measure_gradients=False)
        # for plotting purposes
        emitter_loc = (measures[0][0], measures[0][1])

        best = max(best, max(objs))

        scheduler.tell_dqd(objs, measures, jacobian)

        # sample a batch of branched solution points and evaluate their f and m
        branched_sols = scheduler.ask()
        branched_agents = [Actor(cfg, obs_shape, action_shape).deserialize(sol).to(device) for sol in branched_sols]
        ppo.agents = branched_agents
        objs, measures, metadata = ppo.evaluate(ppo.vec_inference, ppo.multi_eval_env)

        if cfg.weight_decay:
            reg_loss = cfg.weight_decay * np.array([np.linalg.norm(sol) for sol in branched_sols]).reshape(objs.shape)
            objs -= reg_loss

        best = max(best, max(objs))
        scheduler.tell(objs, measures)

        mean_grad_coeffs = scheduler.emitters[0].opt.mu  # keep track of where the emitter is taking us
        mean_grad_coeffs = np.expand_dims(mean_grad_coeffs, axis=0).astype(np.float32)
        log.info(f'New mean coefficients: {mean_grad_coeffs}')
        # for plotting purposes. Forces is determined as the value of the grad coefficients relative to the
        # previous iteration. If they increased, then the emitter should move in the + dir, otherwise the - dir
        delta_grad_coeffs = mean_grad_coeffs - prev_mean_grad_coeffs
        force = (np.sign(delta_grad_coeffs[0][1]), np.sign(delta_grad_coeffs[0][2]))
        prev_mean_grad_coeffs = mean_grad_coeffs

        # now we step towards the new mean coefficient point with ppo and hack dqd to throw away the recombination
        # solution point
        if all(mean_grad_coeffs[0]) == 0:  # this is usually only the case on startup or on cma-es restart
            mean_grad_coeffs[0][0] = 1
        ppo.grad_coeffs = mean_grad_coeffs
        mean_agents[0].reward_normalizer = NormalizeReward(cfg.num_envs)
        ppo.agents = mean_agents
        log.info('Moving the mean solution point...')
        ppo.train(num_updates=5, rollout_length=cfg.rollout_length, calculate_dqd_gradients=False, move_mean_agent=True)
        trained_mean_agent = ppo.agents[0]  # TODO: make this work for multiple emitters?
        # hack the dqd algorithm to make the new mean solution point the one given by ppo rather than the
        # one given by recombination
        new_mean_sol = trained_mean_agent.serialize()
        scheduler.emitters[0].update_theta(new_mean_sol)

        # logging
        log.debug(f'{itr=}, {itrs=}, Progress: {(100.0 * (itr / itrs)):.2f}%')

        if cfg.num_dims <= 2:
            save_heatmap(result_archive, os.path.join(str(heatmap_dir), f'heatmap_{itr:05d}.png'),
                         emitter_loc=emitter_loc, forces=force)

        # Save the archive at the given frequency.
        # Always save on the final iteration.
        final_itr = itr == itrs
        if (itr > 0 and itr % log_arch_freq == 0) or final_itr:
            final_cp_dir = os.path.join(cp_dir, f'cp_{itr:08d}')
            if not os.path.exists(final_cp_dir):
                os.mkdir(final_cp_dir)
            # Save a full archive for analysis.
            df = result_archive.as_pandas(include_solutions=True)
            df.to_pickle(os.path.join(final_cp_dir, f"archive_{itr:08d}.pkl"))
            # save the scheduler for checkpointing
            with open(os.path.join(final_cp_dir, f'scheduler_{itr:08d}.pkl'), 'wb') as f:
                pickle.dump(scheduler, f)

            # save the top 3 checkpoints, delete older ones
            while len(get_checkpoints(str(cp_dir))) > 3:
                oldest_checkpoint = get_checkpoints(str(cp_dir))[0]
                if os.path.exists(oldest_checkpoint):
                    log.info(f'Removing checkpoint {oldest_checkpoint}')
                    shutil.rmtree(oldest_checkpoint)

            # Update the summary statistics for the archive
            if (itr > 0 and itr % log_freq == 0) or final_itr:
                with open(summary_filename, 'a') as summary_file:
                    writer = csv.writer(summary_file)
                    data = [itr, result_archive.stats.qd_score, result_archive.stats.coverage,
                            result_archive.stats.obj_max, result_archive.stats.obj_mean]
                    writer.writerow(data)

        if use_wandb:
            with torch.no_grad():
                normA = torch.linalg.norm(scheduler.emitters[0].opt.A).cpu().numpy().item()
            wandb.log({
                "QD/QD Score": result_archive.stats.qd_score,
                "QD/average performance": result_archive.stats.obj_mean,
                "QD/coverage (%)": result_archive.stats.coverage * 100.0,
                "QD/best score": result_archive.stats.obj_max,
                "QD/iteration": itr,
                "QD/restarts": scheduler.emitters[0].restarts,
                'QD/mean_coeff_obj': mean_grad_coeffs[0][0],
                'XNES/norm_A': normA
            })
            for i in range(1, cfg.num_dims + 1):
                wandb.log({
                    'QD/iteration': itr,
                    f'QD/mean_coeff_measure{i}': mean_grad_coeffs[0][i]
                })


def ant_main(cfg,
             algorithm,
             ppo,
             dim=1000,
             init_pop=100,
             itrs=10000,
             outdir="logs",
             log_freq=1,
             log_arch_freq=10,
             seed=None,
             use_wandb=False):
    """Experimental tool for the mujoco mujoco experiments.

    Args:
        algorithm (str): Name of the algorithm.
        trials (int): Number of experimental trials to run.
        dim (int): Dimensionality of solutions.
        init_pop (int): Initial population size for MAP-Elites (ignored for CMA variants).
        itrs (int): Iterations to run.
        outdir (str): Directory to save output.
        log_freq (int): Number of iterations between computing QD metrics and updating logs.
        log_arch_freq (int): Number of iterations between saving an archive and generating heatmaps.
        seed (int): Seed for the algorithm. By default, there is no seed.
    """
    # Create a shared logging directory for the experiments for this algorithm.
    s_logdir = os.path.join(outdir, f"{algorithm}")
    logdir = Path(s_logdir)
    cfg.logdir = logdir
    outdir = Path(outdir)
    if not outdir.is_dir():
        outdir.mkdir()
    if not logdir.is_dir():
        logdir.mkdir()

    run_experiment(cfg,
                   algorithm,
                   ppo,
                   itrs,
                   outdir,
                   log_freq,
                   log_arch_freq,
                   seed,
                   use_wandb)


if __name__ == '__main__':
    cfg = parse_args()
    cfg.num_emitters = 1

    if cfg.env_type == 'brax':
        vec_env = make_vec_env_brax(cfg)
        cfg.batch_size = int(cfg.env_batch_size * cfg.rollout_length)
        cfg.num_envs = int(cfg.env_batch_size)
    else:
        raise NotImplementedError(f'{cfg.env_type} is undefined for "env_type"')

    cfg.minibatch_size = int(cfg.batch_size // cfg.num_minibatches)
    cfg.obs_shape = vec_env.single_observation_space.shape
    cfg.action_dim = vec_env.single_action_space.shape

    ppo = PPO(seed=cfg.seed, cfg=cfg, vec_env=vec_env)
    if cfg.use_wandb:
        config_wandb(batch_size=cfg.batch_size, total_steps=cfg.total_timesteps, run_name=cfg.wandb_run_name,
                     wandb_group=cfg.wandb_group)
    outdir = cfg.logdir
    assert not os.path.exists(outdir) or cfg.load_scheduler_from_cp is not None,\
        "Warning: this dir exists. Danger of overwriting previous run"
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    cfg.obs_shape = ppo.vec_env.single_observation_space.shape
    cfg.action_shape = ppo.vec_env.single_action_space.shape
    dummy_agent_params = Actor(cfg, cfg.obs_shape, cfg.action_shape).serialize()
    dims = len(dummy_agent_params)

    ant_main(
        cfg,
        algorithm=cfg.dqd_algorithm,
        ppo=ppo,
        dim=dims,
        init_pop=1,
        itrs=cfg.total_iterations,
        outdir=outdir,
        log_arch_freq=cfg.log_arch_freq,
        seed=0,
        use_wandb=cfg.use_wandb
    )
    sys.exit(0)
