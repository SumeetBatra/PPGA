import gym
import numpy as np
import pickle
import pandas

from attrdict import AttrDict
from models.actor_critic import Actor
"""
253, 280, 181, 213, 171
"""


def load_scheduler(scheduler_path):
    with open(scheduler_path, 'rb') as f:
        scheduler = pickle.load(f)
    pass


if __name__ == '__main__':
    # cfg = {'env_name': 'humanoid', 'env_batch_size': None, 'normalize_obs': False, 'normalize_rewards': True,
    #        'num_dims': 2, 'envs_per_model': 1, 'seed': 0, 'num_envs': 1}
    # cfg = AttrDict(cfg)
    # archive_path = '/home/sumeet/QDPPO/experiments/debug/1111/checkpoints/cp_00002000/archive_00002000.pkl'
    # with open(archive_path, 'rb') as f:
    #     archive_df = pickle.load(f)
    # elites = archive_df.query("objective > 2000").sort_values("objective", ascending=False)
    # elites.query('measure_0 > 0.7').query('measure_1 < 0.2').query('measure_2 < 0.2').query('measure_3 < 0.2')
    # agent_params = elites.query('0').to_numpy()[6:]
    # agent = Actor(cfg, obs_shape=(87,), action_shape=(8,)).deserialize(agent_params)
    scheduler_path = 'experiments/paper_qdppo_halfcheetah/1111/checkpoints/cp_00002000/scheduler_00002000.pkl'
    load_scheduler(scheduler_path)
    pass
