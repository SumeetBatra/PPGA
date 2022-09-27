import torch
import copy
import torch.nn as nn
import numpy as np

from utils.policy import Policy


class VectorizedLinearBlock(nn.Module):
    def __init__(self, weights: torch.Tensor, biases=None, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.device = torch.device('cuda')  # TODO: fix this
        self.weight = nn.Parameter(weights).to(self.device)  # one slice of all the mlps we want to process as a batch
        self.bias = nn.Parameter(biases).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        obs_per_weight = x.shape[0] // self.weight.shape[0]
        x = torch.reshape(x, (-1, obs_per_weight, x.shape[1]))
        w_t = torch.transpose(self.weight, 1, 2).to(torch.device('cuda'))  # TODO: fix this
        y = torch.bmm(x, w_t)
        if self.bias is not None:
            y = torch.transpose(y, 0, 1)
            y += self.bias

        out_features = self.weight.shape[1]
        y = torch.reshape(y, shape=(-1, out_features))
        return y


class VectorizedPolicy(Policy):
    def __init__(self, models, model_fn, **kwargs):
        Policy.__init__(self)
        if not isinstance(models, np.ndarray):
            models = np.array(models)
        self.device = torch.device('cuda')  # TODO: fix this
        self.num_models = len(models)
        self.model_fn = model_fn
        self.blocks = self._vectorize_models(models)
        self.layers = nn.Sequential(*self.blocks)
        self.action_logstds = None
        if hasattr(models[0], 'actor_logstd'):
            action_logprobs = [model.actor_logstd for model in models]
            self.action_logstds = nn.Parameter(torch.cat(action_logprobs)).to(self.device)
        self.kwargs = kwargs

    def _vectorize_models(self, models):
        num_layers = len(models[0].layers)
        blocks = []
        for i in range(0, num_layers):
            if not isinstance(models[0].layers[i], nn.Linear):
                continue
            weights_slice = [models[j].layers[i].weight.to(self.device) for j in range(self.num_models)]
            bias_slice = [models[j].layers[i].bias.to(self.device) for j in range(self.num_models)]

            weights_slice = torch.stack(weights_slice)
            bias_slice = torch.stack(bias_slice)
            nonlinear = models[0].layers[i + 1] if i + 1 < num_layers else None
            block = VectorizedLinearBlock(weights_slice, bias_slice)
            blocks.append(block)
            if nonlinear is not None:
                blocks.append(nonlinear)
        return blocks

    @property
    def actor(self):  # TODO: refactor this into a subclass
        return self.layers

    def models_list(self):
        '''
        Returns a list of models view of the object
        '''
        models = [self.model_fn(**self.kwargs) for _ in range(self.num_models)]
        for l, layer in enumerate(self.layers):
            # layer could be a nonlinearity
            if not isinstance(layer, VectorizedLinearBlock):
                continue
            for i in range(len(models)):
                models[i].layers[l].weight.data = layer.weight.data[i]
                models[i].layers[l].bias.data = layer.bias.data[i]

        # update the action_stddevs if they were passed in
        if self.action_logstds is not None:
            for model, logstd in zip(models, self.action_logstds):
                model.actor_logstd.data = logstd.data
        return models

    def forward(self, x):
        return self.layers(x)
