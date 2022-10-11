import gym
import torch
import torch.nn as nn


# taken from https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py
class RunningMeanStd(nn.Module):
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        super().__init__()
        # TODO: these should be float64. Fix this
        self.register_buffer('mean', torch.zeros(shape, dtype=torch.float32))
        self.register_buffer('var', torch.ones(shape, dtype=torch.float32))
        self.count = epsilon

    def update(self, x):
        """ update from a batch of samples"""
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]
        self.mean, self.var, self.count = self.update_from_moments(batch_mean, batch_var, batch_count)
        self.get_buffer('mean')[:] = self.mean
        self.get_buffer('var')[:] = self.var

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        batch_mean = batch_mean.to(self.mean.device)
        batch_var = batch_var.to(self.var.device)
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count


class NormalizeObservation(nn.Module):
    """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.
    Note:
        The normalization depends on past trajectories and observations
        will not be normalized correctly if the wrapper was
        newly instantiated or the policy was changed recently.
    """

    def __init__(self, obs_space_shape, epsilon: float = 1e-8):
        """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.
        Args:
            env (Env): The environment to apply the wrapper
            epsilon: A stability parameter that is used when scaling the observations.
        """
        super(NormalizeObservation, self).__init__()
        self.obs_rms = RunningMeanStd(shape=obs_space_shape)
        self.epsilon = epsilon

    def forward(self, obs):
        obs = self.normalize(obs)
        return obs

    def normalize(self, obs):
        """Normalises the observation using the running mean and variance of the observations."""
        obs = obs.to(self.obs_rms.mean.device)
        self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + self.epsilon)


class NormalizeReward(nn.Module):
    r"""This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.
    The exponential moving average will have variance :math:`(1 - \gamma)^2`.
    Note:
        The scaling depends on past trajectories and rewards will not be scaled correctly if the wrapper was newly
        instantiated or the policy was changed recently.
    """

    def __init__(self, num_envs, gamma: float = 0.99, epsilon: float = 1e-8):
        super(NormalizeReward, self).__init__()
        self.num_envs = num_envs
        self.return_rms = RunningMeanStd(shape=())
        self.returns = torch.zeros((self.num_envs, 1))
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, rews, dones):
        self.returns = self.returns * self.gamma + rews.reshape(self.returns.shape)
        rews = self.normalize(rews)
        self.returns[dones.long()] = 0.0
        return rews

    def normalize(self, rews):
        """Normalizes the rewards with the running mean rewards and their variance."""
        rews = rews.to(self.return_rms.var.device)
        self.return_rms.update(self.returns)
        return rews / torch.sqrt(self.return_rms.var + self.epsilon)