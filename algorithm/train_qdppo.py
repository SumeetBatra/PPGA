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
from utils.utilities import log, config_wandb, get_checkpoints, set_file_handler
from models.actor_critic import Actor
from envs.brax_custom.brax_env import make_vec_env_brax
from utils.normalize_obs import NormalizeReward
from utils.utilities import save_cfg
from utils.archive_utils import save_heatmap, load_scheduler_from_checkpoint
from envs.brax_custom import reward_offset


def parse_args():
    parser = argparse.ArgumentParser()
    #PPO params
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--torch_deterministic", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--use_wandb", default=False, type=lambda x: bool(strtobool(x)),
                        help='Use weights and biases to track the exp')
    parser.add_argument('--wandb_run_name', type=str, default='ppo_ant')
    parser.add_argument('--wandb_group', type=str)

    # args for brax
    parser.add_argument('--env_batch_size', default=1, type=int, help='Number of parallel environments to run')

    # ppo hyperparams
    parser.add_argument('--report_interval', type=int, default=5, help='Log objective results every N updates')
    parser.add_argument('--rollout_length', type=int, default=2048,
                        help='the number of steps to run in each environment per policy rollout')
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
    parser.add_argument("--num_emitters", type=int, default=1, help="Number of parallel"
                                                                    " CMA-ES instances exploring the archive")
    parser.add_argument('--grid_size', type=int, help='Number of cells per archive dimension')
    parser.add_argument("--num_dims", type=int, help="Dimensionality of measures")
    parser.add_argument("--popsize", type=int, required=True, help="Branching factor for each step of MEGA i.e. the number of branching solutions from the current solution point")
    parser.add_argument('--log_arch_freq', type=int, default=10, help='Frequency in num iterations at which we checkpoint the archive')
    parser.add_argument('--load_scheduler_from_cp', type=str, default=None, help='Load an existing QD scheduler from a checkpoint path')
    parser.add_argument('--total_iterations', type=int, default=100, help='Number of iterations to run the entire dqd-rl loop')
    parser.add_argument('--dqd_algorithm', type=str, choices=['cma_mega_adam', 'cma_maega'], help='Which DQD algorithm should be running in the outer loop')
    parser.add_argument('--logdir', type=str, help='Experiment results directory')
    parser.add_argument('--save_heatmaps', type=lambda x: bool(strtobool(x)), default=True, help='Save the archive heatmaps. Only applies to archives with <= 2 measures')
    parser.add_argument('--use_surrogate_archive', type=lambda x: bool(strtobool(x)), default=False, help="Use a surrogate archive at a higher resolution to get a better gradient signal for DQD")
    parser.add_argument('--sigma0', type=float, default=1.0, help='Initial standard deviation parameter for the covariance matrix used in NES methods')
    parser.add_argument('--restart_rule', type=str, choices=['basic', 'no_improvement'])
    parser.add_argument('--calc_gradient_iters', type=int, help='Number of iters to run PPO when estimating the objective-measure gradients (N1)')
    parser.add_argument('--move_mean_iters', type=int, help='Number of iterations to run PPO when moving the mean solution point (N2)')
    parser.add_argument('--archive_lr', type=float, help='Archive learning rate for MAEGA')
    parser.add_argument('--threshold_min', type=float, default=0.0, help='Min objective threshold for adding new solutions to the archive')
    parser.add_argument('--take_archive_snapshots', type=lambda x: bool(strtobool(x)), default=False, help='Log the objective scores in every cell in the archive every log_freq iterations. Useful for pretty visualizations')

    args = parser.parse_args()
    cfg = AttrDict(vars(args))
    return cfg


def create_scheduler(cfg: AttrDict,
                     algorithm: str,
                     seed: int,
                     num_emitters: int,
                     learning_rate: float = None,
                     use_result_archive: bool = True,
                     initial_sol: np.ndarray = None):
    """Creates a scheduler based on the algorithm name.

    Args:
        algorithm (str): Name of the algorithm passed into sphere_main.
        solution_dim(int): Dimensionality of the sphere function.
        archive_dims (int): Dimensionality of the archive.
        learning_rate (float): Learning rate of archive.
        use_result_archive (bool): Whether to use a separate archive to store
            the results.
        seed (int): Main seed or the various components.
        initial_sol: initial solution (agent)
    Returns:
        ribs.schedulers.Scheduler: A ribs scheduler for running the algorithm.
    """
    obs_shape, action_shape = cfg.obs_shape, cfg.action_shape
    action_dim, obs_dim = np.prod(action_shape), np.prod(obs_shape)
    log.debug(f'Environment {cfg.env_name}, {action_dim=}, {obs_dim=}')
    batch_size = cfg.popsize
    qd_offset = reward_offset[cfg.env_name]

    if initial_sol is None:
        initial_agent = Actor(cfg, obs_shape, action_shape)
        initial_sol = initial_agent.serialize()
    solution_dim = len(initial_sol)
    mode = 'batch'
    threshold_min = -np.inf

    # TODO: specify bounds and archive_dims somewhere based on what env was passed in
    bounds = [(0.0, 1.0)] * cfg.num_dims
    archive_dims = [cfg.grid_size] * cfg.num_dims

    if algorithm in ["cma_mae", "cma_maega"]:
        threshold_min = cfg.threshold_min

    if learning_rate is None:
        if algorithm in ["cma_mae", "cma_maega"]:
            learning_rate = cfg.archive_lr
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
    use_surrogate_archive = cfg.use_surrogate_archive
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
                sigma0=cfg.sigma0,
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
                                       sigma0=cfg.sigma0,
                                       step_size=0.0,  # not used
                                       ranker="imp",
                                       selection_rule="mu",
                                       grad_opt="adam",
                                       restart_rule=cfg.restart_rule,
                                       bounds=None,
                                       batch_size=batch_size,
                                       seed=s,
                                       use_wandb=cfg.use_wandb) for s in emitter_seeds
        ]

    log.debug(
        f"Created Scheduler for {algorithm} with an archive learning rate of {learning_rate}, "
        f"and add mode {mode}, using solution dim {solution_dim} and archive "
        f"dims {archive_dims}. Min threshold is {threshold_min}. Restart rule is {cfg.restart_rule}")

    return Scheduler(archive, emitters, result_archive, surrogate_archive, add_mode=mode, reward_offset=qd_offset)


def run_experiment(cfg: AttrDict,
                   algorithm: str,
                   ppo: PPO,
                   itrs: int = 10000,
                   outdir: str = "./experiments",
                   log_freq: int = 1,
                   log_arch_freq: int = 1000,
                   seed: int = None,
                   use_wandb: bool = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Create a directory for this specific trial.
    s_logdir = os.path.join(outdir, f"{cfg.seed}")
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

    archive_snapshot_filename = os.path.join(s_logdir, 'archive_snapshots.csv')
    if cfg.take_archive_snapshots:
        if os.path.exists(archive_snapshot_filename):
            os.remove(archive_snapshot_filename)
        num_cells = np.prod(scheduler.archive.dims)
        with open(archive_snapshot_filename, 'w') as archive_snapshot_file:
            row = ['Iteration'] + [f'cell_{i}' for i in range(num_cells)]
            writer = csv.writer(archive_snapshot_file)
            writer.writerow(row)

    result_archive = scheduler.result_archive
    best = 0.0

    obs_shape = ppo.vec_env.single_observation_space.shape
    action_shape = ppo.vec_env.single_action_space.shape

    # save the initial heatmap
    if cfg.num_dims <= 2:
        save_heatmap(result_archive, os.path.join(str(heatmap_dir), f'heatmap_{0:05d}.png'))

    prev_mean_grad_coeffs = scheduler.emitters[0].opt.mu

    starting_iter = scheduler.emitters[0].itrs  # if loading a checkpoint, this will be > 0
    for itr in range(starting_iter, itrs + 1):
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
        objs, measures, jacobian, metadata = ppo.train(num_updates=cfg.calc_gradient_iters,
                                                       rollout_length=cfg.rollout_length,
                                                       calculate_dqd_gradients=True,
                                                       negative_measure_gradients=False)
        # for plotting purposes
        emitter_loc = (measures[0][0], measures[0][1])

        best = max(best, max(objs))

        scheduler.tell_dqd(objs, measures, jacobian, metadata)

        # sample a batch of branched solution points and evaluate their f and m
        branched_sols = scheduler.ask()
        branched_agents = [Actor(cfg, obs_shape, action_shape).deserialize(sol).to(device) for sol in branched_sols]
        ppo.agents = branched_agents
        objs, measures, metadata = ppo.evaluate(ppo.vec_inference, ppo.vec_env)

        if cfg.weight_decay:
            reg_loss = cfg.weight_decay * np.array([np.linalg.norm(sol) for sol in branched_sols]).reshape(objs.shape)
            objs -= reg_loss

        best = max(best, max(objs))
        restarted = scheduler.tell(objs, measures, metadata)
        if restarted:
            log.debug("Emitter restarted. Changing the mean agent...")
            mean_soln_point = scheduler.emitters[0].theta
            mean_agents = [Actor(cfg, obs_shape, action_shape).deserialize(mean_soln_point).to(device)]
            for agent in mean_agents:
                agent.actor_logstd = torch.nn.Parameter(torch.zeros(1, np.prod(cfg.action_shape)))



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
        ppo.train(num_updates=cfg.move_mean_iters,
                  rollout_length=cfg.rollout_length,
                  calculate_dqd_gradients=False,
                  move_mean_agent=True)
        trained_mean_agent = ppo.agents[0]
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

        if (itr > 0 and itr % log_freq == 0 and cfg.take_archive_snapshots) or (final_itr and cfg.take_archive_snapshots):
            with open(archive_snapshot_filename, 'a') as archive_snapshot_file:
                writer = csv.writer(archive_snapshot_file)
                num_cells = np.prod(scheduler.result_archive.dims)
                elite_scores = [0 for _ in range(num_cells)]
                for elite in scheduler.result_archive:
                    score, index = elite.objective, elite.index
                    elite_scores[index] = score
                data = [itr] + elite_scores
                writer.writerow(data)

        if use_wandb:
            with torch.no_grad():
                normA = torch.linalg.norm(scheduler.emitters[0].opt.A).cpu().numpy().item()
            wandb.log({
                "QD/QD Score": scheduler.archive.stats.qd_score,  # use regular archive for qd score because it factors in the reward offset
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


def qdppo_main(cfg: AttrDict,
               algorithm: str,
               ppo: PPO,
               itrs: int = 10000,
               outdir: str = "logs",
               log_freq: int = 1,
               log_arch_freq: int = 10,
               seed: int = None,
               use_wandb: str = False):
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
    shared_exp_dir = os.path.join(outdir, f"{cfg.seed}")
    exp_dir = Path(shared_exp_dir)
    cfg.logdir = exp_dir
    outdir = Path(outdir)
    if not outdir.is_dir():
        outdir.mkdir()
    if not exp_dir.is_dir():
        exp_dir.mkdir()
    logdir = os.path.join(shared_exp_dir, 'logs')
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    set_file_handler(logdir)

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

    vec_env = make_vec_env_brax(cfg)
    cfg.batch_size = int(cfg.env_batch_size * cfg.rollout_length)
    cfg.num_envs = int(cfg.env_batch_size)

    cfg.minibatch_size = int(cfg.batch_size // cfg.num_minibatches)
    cfg.obs_shape = vec_env.single_observation_space.shape
    cfg.action_dim = vec_env.single_action_space.shape

    ppo = PPO(seed=cfg.seed, cfg=cfg, vec_env=vec_env)
    if cfg.use_wandb:
        config_wandb(batch_size=cfg.batch_size, total_iters=cfg.total_iterations, run_name=cfg.wandb_run_name,
                     wandb_group=cfg.wandb_group)
    outdir = cfg.logdir
    assert not os.path.exists(outdir) or cfg.load_scheduler_from_cp is not None,\
        "Warning: this dir exists. Danger of overwriting previous run"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    cfg.obs_shape = ppo.vec_env.single_observation_space.shape
    cfg.action_shape = ppo.vec_env.single_action_space.shape
    dummy_agent_params = Actor(cfg, cfg.obs_shape, cfg.action_shape).serialize()
    dims = len(dummy_agent_params)

    qdppo_main(
        cfg,
        algorithm=cfg.dqd_algorithm,
        ppo=ppo,
        itrs=cfg.total_iterations,
        outdir=outdir,
        log_arch_freq=cfg.log_arch_freq,
        seed=cfg.seed,
        use_wandb=cfg.use_wandb
    )
    sys.exit(0)
