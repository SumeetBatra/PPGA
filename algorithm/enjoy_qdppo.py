import gym
import numpy as np
import pickle
import pandas

from attrdict import AttrDict
from models.actor_critic import Actor
"""
253, 280, 181, 213, 171
"""

if __name__ == '__main__':
    cfg = {'env_name': 'ant', 'env_batch_size': None, 'normalize_obs': False, 'normalize_rewards': True,
           'num_dims': 4, 'envs_per_model': 1, 'seed': 0, 'num_envs': 1}
    cfg = AttrDict(cfg)
    archive_path = '/home/sumeet/QDPPO/logs/method3_humanoid_ppo20_lr0.3/cma_maega/trial_0/checkpoints/cp_00000970/archive_00000970.pkl'
    with open(archive_path, 'rb') as f:
        archive_df = pickle.load(f)
    elites = archive_df.query("objective > 2000").sort_values("objective", ascending=False)
    elites.query('measure_0 > 0.7').query('measure_1 < 0.2').query('measure_2 < 0.2').query('measure_3 < 0.2')
    agent_params = elites.query('0').to_numpy()[6:]
    agent = Actor(cfg, obs_shape=(87,), action_shape=(8,)).deserialize(agent_params)
    pass
