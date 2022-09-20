import numpy as np
import torch

from time import time
from envs.VecEnv import VecEnv
from utils.utils import log


def try_vec_env():
    num_workers = 4
    envs_per_worker = 1
    vec_env = VecEnv('QDAntBulletEnv-v0', num_workers=num_workers, envs_per_worker=envs_per_worker)
    obs_dim = vec_env.obs_dim
    rand_act = np.random.randn(8)
    vec_env.reset()
    obs, rew, done = vec_env.step(rand_act)
    log.debug(f'{obs=} \n {rew=} \n {done=}')
    log.debug(f'obs shape: {obs.shape}')
    assert obs.shape == torch.Size([num_workers, envs_per_worker, obs_dim + 2])


# def test_throughput():
#     num_workers = 32
#     envs_per_worker = 1
#     vec_env = VecEnv('QDAntBulletEnv-v0', num_workers=num_workers, envs_per_worker=1)
#     vec_env.reset()
#     num_steps = 1000
#     all_obs = []
#     start_time = time()
#     for _ in range(num_steps):
#         rand_act = np.random.randn(8)
#         obs = vec_env.step(rand_act)
#         all_obs.append(obs)
#     elapsed = time() - start_time
#     total_env_steps = num_steps * num_workers * envs_per_worker
#     fps = total_env_steps / elapsed
#     log.debug(f'{fps=}')
#     all_obs = torch.cat(all_obs)
#     log.debug(f"Total obs collected: {all_obs.shape[0]}")


if __name__ == '__main__':
    try_vec_env()
