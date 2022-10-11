import gym
import QDgym
import numpy as np
from envs.wrappers.normalize_numpy import NormalizeObservation, NormalizeReward
from envs.wrappers.reward import ForwardReward, TotalReward, QDReward, QDRLReward


def make_env(env_id, seed, gamma, measure_coeffs=None):
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
        # # TODO: remove this wrapper when done testing
        # if measure_coeffs:
        #     env = QDRLReward(env, measure_coeffs)
        return env
    return thunk
