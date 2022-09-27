import torch
import torch.nn as nn
import gym

from abc import ABC, abstractmethod
from torch.distributions import MultivariateNormal, Categorical


class Policy(ABC, nn.Module):
    def __init__(self):
        super().__init__()
        self.layers: nn.Sequential

    @abstractmethod
    def forward(self, obs):
        pass

    def get_action(self, obs, action_logstds, actions=None):
        means = self.forward(obs)
        action_stds = torch.exp(action_logstds)
        repeats = len(obs) // len(action_stds)
        action_stds = torch.repeat_interleave(action_stds, repeats=repeats, dim=0)
        cov_mat = torch.diag_embed(action_stds)
        probs = torch.distributions.MultivariateNormal(means, cov_mat)
        if actions is None:
            actions = probs.sample()
        return actions, probs.log_prob(actions), probs.entropy()

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
