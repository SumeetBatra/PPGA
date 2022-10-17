import functools
import gym
import brax

from envs import brax_custom
from brax.envs import to_torch
from jax.dlpack import to_dlpack
from envs.wrappers.reward import ForwardReward, TotalReward, QDReward, QDRLReward

import torch
v = torch.ones(1, device='cuda')  # init torch cuda before jax

_to_brax_env_name = {
    'ant': 'brax_custom-ant-v0'
}


def make_vec_env_brax(cfg):
    entry_point = functools.partial(brax_custom.create_gym_env, env_name=cfg.env_name)
    if 'brax_custom-ant-v0' not in gym.envs.registry.env_specs:
        gym.register('brax_custom-ant-v0', entry_point=entry_point)

    vec_env = gym.make(_to_brax_env_name[cfg.env_name], batch_size=cfg.env_batch_size, seed=cfg.seed)
    vec_env = to_torch.JaxToTorchWrapper(vec_env, device='cuda')

    vec_env = gym.wrappers.ClipAction(vec_env)
    vec_env = gym.wrappers.TransformObservation(vec_env, lambda obs: torch.clip(obs, -10, 10))
    vec_env = gym.wrappers.TransformReward(vec_env, lambda reward: torch.clip(reward, -10, 10))
    vec_env = TotalReward(vec_env)

    return vec_env
