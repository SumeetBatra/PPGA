import sys
import gym
import QDgym
import numpy as np
import torch
from attrdict import AttrDict
from RL.ppo import *
from utils.utils import log
from utils.vectorized2 import VectorizedPolicy, VectorizedActorCriticShared
from envs.env import make_env
from QDgym.QDgym_envs import QDAntBulletEnv


def enjoy():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # env = gym.make("QDAntBulletEnv-v0", render=True)
    # env.seed(8679)
    env = make_env('QDAntBulletEnv-v0', seed=0, gamma=0.99)()
    env.render()
    obs_shape, action_shape = env.observation_space.shape, env.action_space.shape
    cp_path = "checkpoints/checkpoint0"
    model_state_dict = torch.load(cp_path)['model_state_dict']
    cfg = {'num_workers': 1, 'envs_per_worker': 1, 'normalize_obs': True, 'normalize_rewards': True}
    cfg = AttrDict(cfg)
    agent = ActorCriticShared(cfg, obs_shape, action_shape).to(device)
    # agent = VectorizedActorCriticShared([agent], ActorCriticShared)
    agent.load_state_dict(model_state_dict)
    obs_mean, obs_var = agent.obs_normalizer.obs_rms.mean, agent.obs_normalizer.obs_rms.var

    obs = env.reset()
    obs = torch.from_numpy(obs).to(device).reshape(1, -1)
    done = False
    total_reward = 0
    while not done:
        # log.debug(f'{done=}')
        obs = (obs - obs_mean) / torch.sqrt(obs_var + 1e-8)
        act, _, _ = agent.get_action(obs)
        # act = agent(obs)
        act = act.squeeze()
        obs, rew, done, _ = env.step(act.detach().cpu().numpy())
        log.debug(f'{rew=}')
        total_reward += rew
        obs = torch.from_numpy(obs).to(device).reshape(1, -1)
        # env.render()
    fitness = total_reward
    log.debug(f'{fitness=}')
    env.close()


if __name__ == '__main__':
    enjoy()
