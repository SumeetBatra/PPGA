import gym
import QDgym
import numpy as np
from envs.wrappers.normalize_numpy import NormalizeObservation, NormalizeReward
from envs.wrappers.reward import ForwardReward, TotalReward, QDReward


def make_env(env_id, seed, gamma):
    def thunk():
        env = gym.make(env_id)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env = TotalReward(env)
        env = QDReward(env)
        return env
    return thunk
