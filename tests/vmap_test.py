import torch
import torch.nn as nn
import numpy as np
import time
from functorch import vmap
from functorch import combine_state_for_ensemble
from models.vectorized import BatchMLP
from functools import partial


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_shape, action_shape: np.ndarray):
        super().__init__()
        self.id = None
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        self.layers = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(action_shape)), std=0.01),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_shape)))

    @property
    def action_log_std(self):
        return self.actor_logstd

    @action_log_std.setter
    def action_log_std(self, log_stddev):
        self.actor_logstd = log_stddev

    def forward(self, x):
        return self.layers(x)


def test_vmap():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_steps = 10000
    batch_traj = torch.randn((5, 8, num_steps)).to(device)
    obs_shape = (8,)
    action_shape = np.array([2])
    agents = [Agent(obs_shape, action_shape).to(device) for _ in range(5)]
    fmodel, params, buffers = combine_state_for_ensemble(agents)
    start_time = time.time()
    for step in range(num_steps):
        next_obs = batch_traj[:, :, step]
        acts_vmap = vmap(fmodel)(params, buffers, next_obs)
    elapsed = time.time() - start_time
    print(f'Execution of vmap took {elapsed:.2f} seconds')


def test_vectorized():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_steps = 1000
    batch_traj = torch.randn((30, 8, num_steps)).to(device)
    obs_shape = (8,)
    action_shape = np.array([2])
    agents = [Agent(obs_shape, action_shape).to(device) for _ in range(5)]
    vec_models = BatchMLP({'hidden_size': 64}, device, partial(Agent), np.array(agents))

    start_time = time.time()
    for step in range(num_steps):
        next_obs = batch_traj[:, :, step]
        acts = vec_models(next_obs)
    elapsed = time.time() - start_time
    print(f'Execution of BatchMLP took {elapsed:.2f} seconds')


if __name__ == '__main__':
    test_vmap()
