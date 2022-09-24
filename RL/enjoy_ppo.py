import sys
import gym
import QDgym
import numpy as np
import torch
from RL.ppo import Agent, LinearPolicy
from utils.utils import log


def enjoy():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # env = gym.make("QDAntBulletEnv-v0", render=True)
    env = gym.make("LunarLanderContinuous-v2")
    obs_shape, action_shape = env.observation_space.shape, env.action_space.shape
    cp_path = "checkpoints/checkpoint0"
    model_state_dict = torch.load(cp_path)['model_state_dict']
    # agent = Agent(obs_shape, action_shape).to(device)
    agent = LinearPolicy(obs_shape, action_shape).to(device)
    agent.load_state_dict(model_state_dict)

    obs = env.reset()
    obs = torch.from_numpy(obs).to(device).reshape(1, -1)
    done = False
    for _ in range(5000):
        log.debug(f'{done=}')
        act, _, _ = agent.get_action(obs)
        act = act.squeeze()
        obs, _, _, _ = env.step(act.cpu().numpy())
        obs = torch.from_numpy(obs).to(device).reshape(1, -1)
        env.render()
    fitness = env.tot_reward
    log.debug(f'{fitness=}')
    env.close()


if __name__ == '__main__':
    enjoy()
