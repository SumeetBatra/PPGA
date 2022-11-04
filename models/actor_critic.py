from typing import List

import torch
import torch.nn as nn
import numpy as np

from models.policy import StochasticPolicy


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCriticSeparate(StochasticPolicy):
    def __init__(self, obs_shape, action_shape: np.ndarray, **kwargs):
        super().__init__(normalize_obs=kwargs.get('normalize_obs', False), normalize_rewards=kwargs.get('normalize_rewards', False))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(action_shape)), std=0.01)
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        cov_mat = torch.diag_embed(action_std)
        probs = torch.distributions.MultivariateNormal(action_mean, cov_mat)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()


class ActorCriticShared(StochasticPolicy):
    def __init__(self, cfg, obs_shape, action_shape: np.ndarray):
        super().__init__(cfg)

        self.core = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh()
        )

        self.actor_head = nn.Sequential(
            layer_init(nn.Linear(64, np.prod(action_shape)), std=0.01)
        )

        self.critic_head = nn.Sequential(
            layer_init(nn.Linear(64, 1), std=1.0)
        )

        self._actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_shape)))

    @property
    def actor_logstd(self):
        return self._actor_logstd

    def get_action(self, obs, action=None):
        core_out = self.core(obs)
        action_mean = self.actor_head(core_out)
        action_logstd = self._actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        cov_mat = torch.diag_embed(action_std)
        probs = torch.distributions.MultivariateNormal(action_mean, cov_mat)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()

    def get_value(self, obs):
        core_out = self.core(obs)
        value = self.critic_head(core_out)
        return value

    def forward(self, obs):
        '''Gets the raw logits from the actor head. Treats the policy as deterministic'''
        core_out = self.core(obs)
        return self.actor_head(core_out)


class QDActorCriticShared(ActorCriticShared):
    def __init__(self, cfg, obs_shape, action_shape: np.ndarray, num_dims):
        ActorCriticShared.__init__(self, cfg, obs_shape, action_shape)
        self.num_dims = num_dims
        # create new critic heads, one for each measure
        self.measure_critic_heads = nn.Sequential(layer_init(nn.Linear(64, num_dims), std=1.0))
        self.measure_coeffs = torch.zeros(num_dims)  # algorithm should be responsible for changing this

    def get_measure_values(self, obs):
        core_out = self.core(obs)
        value = self.measure_critic_heads(core_out)
        return value


class LinearPolicy(nn.Module):
    def __init__(self, obs_shape, action_shape):
        super().__init__()
        self.actor = layer_init(nn.Linear(np.prod(obs_shape),
                                          np.prod(action_shape)))
        self.layer = nn.Sequential(*[self.actor])
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_shape)))

    def get_action(self, obs, action=None):
        action_mean = self.actor(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        cov_mat = torch.diag_embed(action_std)
        probs = torch.distributions.MultivariateNormal(action_mean, cov_mat)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()

    def forward(self, x):
        return self.actor(x)

    def update_weights(self, weights):
        self.actor.weight.data = weights


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

        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_shape)))

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
                    layer_init(nn.Linear(64, 64)),
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
