import torch
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

from attrdict import AttrDict
from ribs.archives import CVTArchive, GridArchive
from ribs.visualize import cvt_archive_heatmap, grid_archive_heatmap
from typing import Optional
from models.actor_critic import Actor
from models.vectorized import VectorizedActor
from envs.brax_custom.brax_env import make_vec_env_brax


def save_heatmap(archive, heatmap_path, emitter_loc: Optional[tuple[float, ...]] = None,
                 forces: Optional[tuple[float, ...]] = None):
    """Saves a heatmap of the archive to the given path.
    Args:
        archive (GridArchive or CVTArchive): The archive to save.
        heatmap_path: Image path for the heatmap.
        emitter_loc: Where the emitter is in the archive. Determined by the measures of the mean solution point
        force: the direction that the emitter is being pushed towards. Determined by the gradient coefficients of
        the mean solution point
    """
    if isinstance(archive, GridArchive):
        plt.figure(figsize=(8, 6))
        grid_archive_heatmap(archive, emitter_loc=emitter_loc, forces=forces)
        plt.tight_layout()
        plt.savefig(heatmap_path)
    elif isinstance(archive, CVTArchive):
        plt.figure(figsize=(16, 12))
        cvt_archive_heatmap(archive)
        plt.tight_layout()
        plt.savefig(heatmap_path)
    plt.close('all')


def load_scheduler_from_checkpoint(scheduler_path):
    assert os.path.exists(scheduler_path), f'Error! {scheduler_path=} does not exist'
    with open(scheduler_path, 'rb') as f:
        scheduler = pickle.load(f)
    return scheduler


def load_archive(archive_path):
    assert os.path.exists(archive_path), f'Error! {archive_path=} does not exist'
    with open(archive_path, 'rb') as f:
        archive = pickle.load(f)
    return archive


def evaluate(vec_agent, vec_env, num_dims):
    '''
    Evaluate all agents for one episode
    :param vec_agent: Vectorized agents for vectorized inference
    :returns: Sum rewards and measures for all agents
    '''

    total_reward = np.zeros(vec_env.num_envs)
    traj_length = 0
    num_steps = 1000
    device = torch.device('cuda')

    obs = vec_env.reset()
    obs = obs.to(device)
    dones = torch.BoolTensor([False for _ in range(vec_env.num_envs)])
    all_dones = torch.zeros((num_steps, vec_env.num_envs)).to(device)
    measures_acc = torch.zeros((num_steps, vec_env.num_envs, num_dims)).to(device)
    measures = torch.zeros((vec_env.num_envs, num_dims)).to(device)

    while not torch.all(dones):
        with torch.no_grad():
            # acts, _, _ = vec_agent.get_action(obs)
            acts = vec_agent(obs)
            obs, rew, next_dones, infos = vec_env.step(acts)
            measures_acc[traj_length] = infos['measures']
            obs = obs.to(device)
            total_reward += rew.detach().cpu().numpy() * ~dones.cpu().numpy()
            dones = torch.logical_or(dones, next_dones.cpu())
            all_dones[traj_length] = dones.long().clone()
            traj_length += 1

    # the first done in each env is where that trajectory ends
    traj_lengths = torch.argmax(all_dones, dim=0)
    # TODO: figure out how to vectorize this
    for i in range(vec_env.num_envs):
        measures[i] = measures_acc[:traj_lengths[i], i].sum(dim=0) / traj_lengths[i]
    measures = measures.reshape(vec_agent.num_models, vec_env.num_envs // vec_agent.num_models, -1).mean(dim=1)

    total_reward = total_reward.reshape((vec_agent.num_models, vec_env.num_envs // vec_agent.num_models))
    total_reward = total_reward.mean(axis=1)
    return total_reward.reshape(-1, ), measures.reshape(-1, num_dims).detach().cpu().numpy()


def reevaluate_archive(archive_df, env_cfg):
    solutions = archive_df.filter(like='solution').to_numpy()
    num_sols = solutions.shape[0]
    env_cfg.env_batch_size = num_sols
    vec_env = make_vec_env_brax(env_cfg)

    agents = []
    cfg = AttrDict({'normalize_obs': False, 'normalize_rewards': True, 'num_envs': num_sols, 'num_dims': env_cfg.num_dims})
    obs_shape, action_shape = vec_env.single_observation_space.shape, vec_env.single_action_space.shape
    device = torch.device('cuda')

    for sol in solutions:
        agent = Actor(cfg, obs_shape=obs_shape, action_shape=action_shape).deserialize(sol).to(device)
        agents.append(agent)
    vec_inference = VectorizedActor(cfg, agents, Actor, obs_shape=obs_shape, action_shape=action_shape).to(device)
    all_objs, all_measures = [], []
    num_iters = 4
    for _ in range(num_iters):
        objs, measures = evaluate(vec_inference, vec_env, env_cfg.num_dims)
        all_objs.append(objs)
        all_measures.append(measures)
    all_objs, all_measures = np.concatenate(all_objs).reshape(num_iters, -1).mean(axis=0),\
                             np.concatenate(all_measures).reshape(num_iters, -1, env_cfg.num_dims).mean(axis=0)

    # create a new archive
    archive_dims = [100, 100]
    bounds = [(0., 1.0) for _ in range(cfg.num_dims)]
    new_archive = GridArchive(solution_dim=solutions.shape[1],
                              dims=archive_dims,
                              ranges=bounds,
                              threshold_min=0.0,
                              seed=env_cfg.seed)
    # add the re-evaluated solutions to the new archive
    new_archive.add(
        solutions,
        all_objs,
        all_measures
    )

    # save a heatmap of the new archive
    experiment_dir = '/home/sbatra/QDPPO/logs/method3_walker2d_debug_pycma/cma_maega/trial_0'
    analysis_dir = os.path.join(experiment_dir, 'post_hoc_analysis')
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)
    heatmap_path = os.path.join(analysis_dir, 'new_archive_heatmap.png')
    save_heatmap(new_archive, heatmap_path)


if __name__ == '__main__':
    archive_path = '/home/sbatra/QDPPO/logs/method3_walker2d_debug_pycma/cma_maega/trial_0/checkpoints/cp_00002000/archive_00002000.pkl'
    archive_df = load_archive(archive_path)
    env_cfg = AttrDict({'env_name': 'walker2d', 'num_dims': 2, 'seed': 1111})
    reevaluate_archive(archive_df, env_cfg)


