import sys
import gym
import QDgym
import numpy as np
import torch
from RL.ppo import Agent, LinearPolicy
from utils.utils import log
from utils.vectorized2 import VectorizedPolicy


def enjoy():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # env = gym.make("QDAntBulletEnv-v0", render=True)
    env = gym.make("LunarLanderContinuous-v2")
    env.seed(0)
    obs_shape, action_shape = env.observation_space.shape, env.action_space.shape
    cp_path = "checkpoints/checkpoint0"
    model_state_dict = torch.load(cp_path)['model_state_dict']
    agent = Agent(obs_shape, action_shape).to(device)
    # agent = VectorizedPolicy([agent], Agent)
    agent.load_state_dict(model_state_dict)

    obs = env.reset()
    obs = torch.from_numpy(obs).to(device).reshape(1, -1)
    done = False
    total_reward = 0
    for _ in range(512):
        log.debug(f'{done=}')
        act, _, _ = agent.get_action(obs, agent.action_logstds)
        act = act.squeeze()
        obs, rew, _, _ = env.step(act.cpu().numpy())
        log.debug(f'{rew=}')
        total_reward += rew
        obs = torch.from_numpy(obs).to(device).reshape(1, -1)
        env.render()
    fitness = total_reward
    log.debug(f'{fitness=}')
    env.close()


if __name__ == '__main__':
    enjoy()
