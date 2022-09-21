import sys
import gym
import QDgym
import numpy as np
import torch
from RL.ppo import Agent
from utils.utils import log


def enjoy():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = gym.make("QDAntBulletEnv-v0", render=True)
    obs_shape, action_shape = env.observation_space.shape, env.action_space.shape
    cp_path = "checkpoints/checkpoint0"
    model_state_dict = torch.load(cp_path)['model_state_dict']
    agent = Agent(obs_shape, action_shape).to(device)
    agent.load_state_dict(model_state_dict)

    obs = env.reset()
    obs = torch.from_numpy(obs).to(device).reshape(1, -1)
    done = False
    for _ in range(2000):
        log.debug(f'{done=}')
        act, _, _, _ = agent.get_action_and_value(obs)
        act = act.squeeze()
        env.step(act.cpu().numpy())
    fitness = env.tot_reward
    log.debug(f'{fitness=}')
    env.close()


if __name__ == '__main__':
    enjoy()
