import torch
import torch.nn as nn
import numpy as np
import gym

from abc import ABC, abstractmethod
from torch.distributions import MultivariateNormal, Categorical
from utils.normalize_obs import NormalizeReward, NormalizeObservation


class StochasticPolicy(ABC, nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers: nn.Sequential

        if cfg.normalize_obs:
            self.obs_normalizer = NormalizeObservation((28,))  # TODO: fix this
        if cfg.normalize_rewards:
            self.reward_normalizer = NormalizeReward(cfg.envs_per_model)

    @abstractmethod
    def forward(self, obs):
        pass

    @abstractmethod
    def get_action(self, obs, action=None):
        ...

    @staticmethod
    def get_action_distribution(action_space, raw_logits, scale=None):
        if isinstance(action_space, gym.spaces.Discrete):
            return Categorical(logits=raw_logits)
        if isinstance(action_space, gym.spaces.Box):
            assert scale is not None, "Must pass in the stddev vector!"
            cov_mat = torch.diag(scale)
            return MultivariateNormal(loc=raw_logits, covariance_matrix=cov_mat)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    # From DQDRL paper https://arxiv.org/pdf/2202.03666.pdf
    def serialize(self):
        '''
        Returns a 1D numpy array view of the entire policy.
        '''
        return np.concatenate(
            [p.data.cpu().detach().numpy().ravel() for p in self.parameters()])

    def deserialize(self, array: np.ndarray):
        '''
        Update the weights of this policy with the weights from the 1D
        array of parameters
        '''
        """Loads parameters from 1D array."""
        array = np.copy(array)
        arr_idx = 0
        for param in self.parameters():
            shape = tuple(param.data.shape)
            length = np.product(shape)
            block = array[arr_idx:arr_idx + length]
            if len(block) != length:
                raise ValueError("Array not long enough!")
            block = np.reshape(block, shape)
            arr_idx += length
            param.data = torch.from_numpy(block).float()
        return self

    def gradient(self):
        '''Returns 1D numpy array view of the gradients of this actor'''
        return np.concatenate(
            [p.grad.cpu().detach().numpy().ravel() for p in self.parameters()]
        )
