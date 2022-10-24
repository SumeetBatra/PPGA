import torch
import torch.nn as nn
import numpy as np

from models.policy import StochasticPolicy
from abc import ABC, abstractmethod
from typing import List
from utils.normalize_obs import NormalizeReward, NormalizeObservation


class VectorizedLinearBlock(nn.Module):
    def __init__(self, weights: torch.Tensor, biases=None, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight = nn.Parameter(weights).to(self.device)  # one slice of all the mlps we want to process as a batch
        self.bias = nn.Parameter(biases).to(self.device) if biases is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        obs_per_weight = x.shape[0] // self.weight.shape[0]
        x = torch.reshape(x, (-1, obs_per_weight, x.shape[1]))
        w_t = torch.transpose(self.weight, 1, 2).to(self.device)
        y = torch.bmm(x, w_t)
        if self.bias is not None:
            y = torch.transpose(y, 0, 1)
            y += self.bias

        out_features = self.weight.shape[1]
        y = torch.transpose(y, 0, 1)
        y = torch.reshape(y, shape=(-1, out_features))
        return y


class VectorizedPolicy(StochasticPolicy, ABC):
    def __init__(self, cfg, models, model_fn, **kwargs):
        StochasticPolicy.__init__(self, cfg)
        if not isinstance(models, np.ndarray):
            models = np.array(models)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_models = len(models)
        self.model_fn = model_fn
        self.blocks: List[VectorizedLinearBlock]
        self.actor_mean: nn.Sequential
        self.actor_logstd: nn.Parameter
        self.kwargs = kwargs

        if cfg.normalize_obs:
            self.obs_normalizers = [model.obs_normalizer for model in models]
        if cfg.normalize_rewards:
            self.rew_normalizers = [model.reward_normalizer for model in models]
            self.measure_normalizers = [model.measure_normalizer for model in models]

    def _vectorize_layers(self, layer_name, models):
        '''
        Vectorize a specific nn.Sequential list of layers across all models of homogenous architecture
        :param layer_name: name of a nn.Sequential block
        :return: vectorized nn.Sequential block
        '''
        assert hasattr(models[0], layer_name), f'{layer_name=} not in the model'
        all_models_layers = [getattr(models[i], layer_name) for i in range(self.num_models)]
        num_layers = len(getattr(models[0], layer_name))
        blocks = []
        for i in range(0, num_layers):
            if not isinstance(all_models_layers[0][i], nn.Linear):
                continue
            weights_slice = [all_models_layers[j][i].weight.to(self.device) for j in range(self.num_models)]
            bias_slice = [all_models_layers[j][i].bias.to(self.device) for j in range(self.num_models)]

            weights_slice = torch.stack(weights_slice)
            bias_slice = torch.stack(bias_slice)
            nonlinear = all_models_layers[0][i + 1] if i + 1 < num_layers else None
            block = VectorizedLinearBlock(weights_slice, bias_slice)
            blocks.append(block)
            if nonlinear is not None:
                blocks.append(nonlinear)
        return blocks

    def vec_to_models(self):
        '''
        Returns a list of models view of the object
        '''
        models = [self.model_fn(cfg=self.cfg, **self.kwargs) for _ in range(self.num_models)]
        for i, model in enumerate(models):
            for l, layer in enumerate(self.actor_mean):
                # layer could be a nonlinearity
                if not isinstance(layer, VectorizedLinearBlock):
                    continue
                model.actor_mean[l].weight.data = layer.weight.data[i]
                model.actor_mean[l].bias.data = layer.bias.data[i]

                # update obs/rew normalizers
                if self.cfg.normalize_obs:
                    model.obs_normalizer = self.obs_normalizers[i]
                if self.cfg.normalize_rewards:
                    model.reward_normalizer = self.rew_normalizers[i]

                # update action logprobs
                model.actor_logstd.data = self.actor_logstd[i]

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def get_action(self, obs, action=None):
        pass

    def vec_normalize_obs(self, obs):
        # TODO: make this properly vectorized
        obs = obs.reshape(self.num_models, obs.shape[0] // self.num_models, -1)
        for i, (model_obs, normalizer) in enumerate(zip(obs, self.obs_normalizers)):
            obs[i] = normalizer(model_obs)
        return obs.reshape(-1, obs.shape[-1])

    def vec_normalize_rewards(self, rewards, next_done):
        # TODO: make this properly vectorized
        envs_per_model = self.cfg.envs_per_model
        rewards = rewards.reshape(self.num_models, envs_per_model)
        next_dones = next_done.reshape(self.num_models, envs_per_model)
        for i, (model_rews, dones, normalizer) in enumerate(zip(rewards, next_dones, self.rew_normalizers)):
            rewards[i] = normalizer(model_rews, dones)
        return rewards.reshape(-1)

    def vec_normalize_measures(self, measures, next_done):
        # TODO: make this properly vectorized
        num_envs = measures.shape[0]
        envs_per_model = num_envs // self.num_models
        measures = measures.reshape(self.num_models, envs_per_model, self.cfg.num_dims)
        next_dones = next_done.reshape(self.num_models, envs_per_model)
        for i, (model_rews, dones, normalizer) in enumerate(zip(measures, next_dones, self.measure_normalizers)):
            measures[i] = normalizer(model_rews, dones)
        return measures.reshape(num_envs, -1)


class VectorizedActor(VectorizedPolicy):
    def __init__(self, cfg, models, model_fn, **kwargs):
        VectorizedPolicy.__init__(self, cfg, models, model_fn, **kwargs)
        self.blocks = self._vectorize_layers('actor_mean', models)
        self.actor_mean = nn.Sequential(*self.blocks)
        action_logprobs = [model.actor_logstd for model in models]
        action_logprobs = torch.cat(action_logprobs).to(self.device)
        self.actor_logstd = nn.Parameter(action_logprobs)

    def forward(self, x):
        return self.actor_mean(x)

    def get_action(self, obs, action=None):
        action_mean = self.actor_mean(obs)
        repeats = obs.shape[0] // self.num_models
        action_logstd = torch.repeat_interleave(self.actor_logstd, repeats, dim=0)
        action_logstd = action_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        cov_mat = torch.diag_embed(action_std)
        probs = torch.distributions.MultivariateNormal(action_mean, cov_mat)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()


class VectorizedActorCriticShared(StochasticPolicy):
    def __init__(self, cfg, models, model_fn, **kwargs):
        StochasticPolicy.__init__(self, cfg)
        if not isinstance(models, np.ndarray):
            models = np.array(models)
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # TODO: fix this
        self.num_models = len(models)
        self.model_fn = model_fn
        self.core_blocks, self.actor_head_blocks, self.critic_head_blocks = self._vectorize_layers('core', models), \
                                                                            self._vectorize_layers('actor_head',
                                                                                                   models), \
                                                                            self._vectorize_layers('critic_head',
                                                                                                   models)
        self.core, self.actor_head, self.critic_head = nn.Sequential(*self.core_blocks), \
                                                       nn.Sequential(*self.actor_head_blocks), \
                                                       nn.Sequential(*self.critic_head_blocks)
        action_logprobs = [model.actor_logstd for model in models]
        action_logprobs = torch.cat(action_logprobs).to(self.device)
        self._actor_logstd = nn.Parameter(action_logprobs)
        self.kwargs = kwargs

        if cfg.normalize_obs:
            self.obs_normalizers = [model.obs_normalizer for model in models]
        if cfg.normalize_rewards:
            self.rew_normalizers = [model.reward_normalizer for model in models]

    def _vectorize_layers(self, layer_name, models):
        '''
        Vectorize a specific nn.Sequential list of layers across all models of homogenous architecture
        :param layer_name: name of a nn.Sequential block
        :return: vectorized nn.Sequential block
        '''
        assert hasattr(models[0], layer_name), f'{layer_name=} not in the model'
        all_models_layers = [getattr(models[i], layer_name) for i in range(self.num_models)]
        num_layers = len(getattr(models[0], layer_name))
        blocks = []
        for i in range(0, num_layers):
            if not isinstance(all_models_layers[0][i], nn.Linear):
                continue
            weights_slice = [all_models_layers[j][i].weight.to(self.device) for j in range(self.num_models)]
            bias_slice = [all_models_layers[j][i].bias.to(self.device) for j in range(self.num_models)]

            weights_slice = torch.stack(weights_slice)
            bias_slice = torch.stack(bias_slice)
            nonlinear = all_models_layers[0][i + 1] if i + 1 < num_layers else None
            block = VectorizedLinearBlock(weights_slice, bias_slice)
            blocks.append(block)
            if nonlinear is not None:
                blocks.append(nonlinear)
        return blocks

    def get_action(self, obs, action=None):
        core_out = self.core(obs)
        action_mean = self.actor_head(core_out)
        repeats = obs.shape[0] // self.num_models
        action_logstd = torch.repeat_interleave(self._actor_logstd, repeats, dim=0)
        action_logstd = action_logstd.expand_as(action_mean)
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

    def vec_normalize_obs(self, obs):
        # TODO: make this properly vectorized
        obs = obs.reshape(self.num_models, obs.shape[0] // self.num_models, -1)
        for i, (model_obs, normalizer) in enumerate(zip(obs, self.obs_normalizers)):
            obs[i] = normalizer(model_obs)
        return obs.reshape(-1, obs.shape[-1])

    def vec_normalize_rewards(self, rewards, next_done):
        # TODO: make this properly vectorized
        envs_per_model = self.cfg.envs_per_model
        rewards = rewards.reshape(self.num_models, envs_per_model)
        next_dones = next_done.reshape(self.num_models, envs_per_model)
        for i, (model_rews, dones, normalizer) in enumerate(zip(rewards, next_dones, self.rew_normalizers)):
            rewards[i] = normalizer(model_rews, dones)
        return rewards.reshape(-1)

    def vec_to_models(self):
        '''
        Inverse operation of _vectorize_layers
        '''
        models = [self.model_fn(cfg=self.cfg, num_dims=self.cfg.num_dims, **self.kwargs) for _ in range(self.num_models)]
        for i, model in enumerate(models):
            # update weights of each model core
            for l, layer in enumerate(self.core):
                # layer could be a nonlinearity
                if not isinstance(layer, VectorizedLinearBlock):
                    continue
                model.core[l].weight.data = layer.weight.data[i]
                model.core[l].bias.data = layer.bias.data[i]

            # update weights of each head
            model.actor_head[0].weight.data = self.actor_head[0].weight.data[i]
            model.actor_head[0].bias.data = self.actor_head[0].bias.data[i]
            model.critic_head[0].weight.data = self.critic_head[0].weight.data[i]
            model.critic_head[0].bias.data = self.critic_head[0].bias.data[i]

            # update the obs/rew normalizers if enabled
            if self.cfg.normalize_obs:
                model.obs_normalizer = self.obs_normalizers[i]
            if self.cfg.normalize_rewards:
                model.reward_normalizer = self.rew_normalizers[i]

            # update the action-logstd params
            model._actor_logstd.data = self._actor_logstd[i]

        return models

