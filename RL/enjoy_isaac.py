import isaacgym
import isaacgymenvs
import torch
import gym
import argparse
from models.actor_critic import Actor
from attrdict import AttrDict
from distutils.util import strtobool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--normalize_obs', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--normalize_rewards', type=lambda x: bool(strtobool(x)), default=False)

    args = parser.parse_args()
    cfg = AttrDict(vars(args))
    return cfg


def enjoy(cfg, render=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = isaacgymenvs.make(
        seed=cfg.seed,
        task=cfg.env_name,
        num_envs=1,
        sim_device='cuda:0',
        rl_device='cuda:0',
        graphics_device_id=0,
        headless=not render,
        force_render=False
    )

    obs_shape = env.obs_space.shape
    action_shape = env.action_space.shape

    model_state_dict = torch.load(cfg.model_path)['model_state_dict']
    model_state_dict['actor_logstd'] = model_state_dict['actor_logstd'].reshape(1, -1)
    agent = Actor(cfg, obs_shape, action_shape).to(device)
    agent.load_state_dict(model_state_dict)

    obs = env.reset()
    total_reward = 0
    done = False
    while not done:
        if render:
            env.render()
        with torch.no_grad():
            act, _, _ = agent.get_action(obs['obs'])
            obs, rew, done, info = env.step(act)
            total_reward += rew
    print(f'{total_reward=}')


if __name__ == '__main__':
    cfg = parse_args()
    cfg.num_envs = 1
    enjoy(cfg, render=True)
