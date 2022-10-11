import torch
import torch.nn as nn
import numpy as np

from models.policy import StochasticPolicy


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_shape, action_shape: np.ndarray):
        super().__init__()

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(action_shape)), std=0.01),
        )

        self._actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_shape)))

    def forward(self, x):
        return self.actor_mean(x)

    def get_action(self, obs, action=None):
        action_mean = self.actor_mean(obs)
        action_logstd = self._actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        cov_mat = torch.diag_embed(action_std)
        probs = torch.distributions.MultivariateNormal(action_mean, cov_mat)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()


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


class GlobalCritic(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()
        self.m_dim = 2  # dimensionality of the measures
        self.critic = nn.Sequential(
            # layer_init(nn.Linear(np.array(obs_shape).prod() + self.m_dim, 64)),
            layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_value(self, obs):
        return self.critic(obs)

