from typing import List

import torch
import torch.nn as nn
import numpy as np

from models.policy import StochasticPolicy


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(StochasticPolicy):
    def __init__(self, cfg, obs_shape, action_shape: np.ndarray):
        super().__init__(cfg)

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(action_shape)), std=0.01),
        )

        self.actor_logstd = torch.zeros(1, np.prod(action_shape))

    def forward(self, x):
        return self.actor_mean(x)

    def get_action(self, obs, action=None):
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = torch.distributions.Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy()


class PGAMEActor(nn.Module):
    def __init__(self, obs_shape, action_shape):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_shape, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, np.prod(action_shape)),
            nn.Tanh()
        )
        self.actor_logstd = -100.0 * torch.ones(action_shape[0])

    def forward(self, obs):
        return self.network(obs)

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


class Critic(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()
        self.core = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_value(self, obs):
        core_out = self.core(obs)
        return self.critic(core_out)

    def forward(self, obs):
        return self.get_value(obs)


class QDCritic2(nn.Module):
    def __init__(self, obs_shape, measure_dim, critics_list: List[nn.Module] = None):
        super(QDCritic2, self).__init__()
        self.measure_dim = measure_dim
        if critics_list is None:
            self.all_critics = nn.ModuleList([
                nn.Sequential(
                    layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
                    nn.Tanh(),
                    layer_init(nn.Linear(64, 128)),
                    nn.Tanh(),
                    layer_init(nn.Linear(128, 64)),
                    nn.Tanh(),
                    layer_init(nn.Linear(64, 1), std=1.0)
                ) for _ in range(measure_dim + 1)
            ])
        else:
            self.all_critics = nn.ModuleList(critics_list)

    def get_value_at(self, obs, dim):
        return self.all_critics[dim](obs)

    def get_all_values(self, obs):
        all_vals = []
        for critic in self.all_critics:
            all_vals.append(critic(obs))
        all_vals = torch.cat(all_vals).to(obs.device)
        return all_vals

    def get_value(self, obs):
        '''
        Implemented for backwards compatibility
        '''
        return self.all_critics[0](obs)


class QDCritic(Critic):
    def __init__(self, obs_shape, measure_dim):
        Critic.__init__(self, obs_shape)
        self.measure_critics = torch.nn.ModuleList([
            nn.Sequential(layer_init(nn.Linear(64, 1), std=1.0)) for _ in range(measure_dim)
        ])

    def get_measure_value(self, obs, dim):
        core_out = self.core(obs)
        return self.measure_critics[dim](core_out)

    def get_obj_and_measure_values(self, obs):
        core_out = self.core(obs)
        obj_val = self.critic(core_out)
        measure_vals = []
        for critic in self.measure_critics:
            measure_vals.append(critic(core_out))
        measure_vals = torch.cat(measure_vals).reshape(-1, self.measure_dim).to(obj_val.device)
        return obj_val, measure_vals


class QDMeasureCritic(nn.Module):
    '''
    QD Critic with measure_critic heads and conditioned on measure coefficients
    '''
    def __init__(self, obs_shape, measure_dim):
        super().__init__()
        # re-write the core layers to include measure coeffs
        self.core = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod() + measure_dim + 1, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )

    def get_value(self, obs):
        return self.core(obs)

    def forward(self, obs):
        return self.core(obs)
