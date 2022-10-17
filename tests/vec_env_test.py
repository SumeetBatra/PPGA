import numpy as np
import gym
from envs.cpu.env import make_env


def test_vec_env():
    num_envs = 1
    envs = gym.vector.SyncVectorEnv([make_env('LunarLanderContinuous-v2', seed=0, gamma=0.99) for _ in range(num_envs)])
    dones = [False for _ in range(num_envs)]
    obs = envs.reset()
    for step in range(1000):
        acts = np.random.randn(2, 2)
        obs, _, done, _ = envs.step(acts)
        print(step)


if __name__ == '__main__':
    test_vec_env()
