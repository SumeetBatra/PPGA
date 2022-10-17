import functools
import time

from IPython.display import HTML, Image
import gym

import brax

from brax import envs
from brax import jumpy as jp
from brax.envs import to_torch
from brax.io import html
from brax.io import image
import jax
from jax import numpy as jnp
from jax.dlpack import to_dlpack
import torch
v = torch.ones(1, device='cuda')  # init torch cuda before jax


def brax_test():
    entry_point = functools.partial(envs.create_gym_env, env_name='ant')
    if 'brax_custom-ant-v0' not in gym.envs.registry.env_specs:
        gym.register('brax_custom-ant-v0', entry_point=entry_point)

    # create a gym environment that contains 4096 parallel ant environments
    gym_env = gym.make("brax_custom-ant-v0", batch_size=1)

    # wrap it to interoperate with torch data structures
    gym_env = to_torch.JaxToTorchWrapper(gym_env, device='cuda')

    # jit compile env.reset
    obs = gym_env.reset()

    # jit compile env.step
    action = torch.rand(gym_env.action_space.shape, device='cuda') * 2 - 1
    obs, reward, done, info = gym_env.step(action)

    before = time.time()

    for _ in range(1000):
        action = torch.rand(gym_env.action_space.shape, device='cuda') * 2 - 1
        obs, rewards, done, info = gym_env.step(action)

    duration = time.time() - before
    print(f'time for {409600} steps: {duration:.2f}s ({int(409600 / duration)} steps/sec)')

    before = time.time()

    for _ in range(100):
        action = torch.rand(gym_env.action_space.shape, device='cuda') * 2 - 1
        obs, rewards, done, info = gym_env.step(action)

    duration = time.time() - before
    print(f'time for {409600} steps: {duration:.2f}s ({int(409600 / duration)} steps/sec)')


def create_qdbrax_gym():
    env = brax.envs._envs['ant'](legacy_spring=True)

    env = brax.envs.wrappers.EpisodeWrapper(env, 1000, 1)
    env = brax.envs.wrappers.VectorWrapper(env, batch_size=4096)
    env = brax.envs.wrappers.AutoResetWrapper(env)
    env = to_torch.JaxToTorchWrapper(env, device='cuda')
    return env


if __name__ == '__main__':
    brax_test()
