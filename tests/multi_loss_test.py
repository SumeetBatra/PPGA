import torch
import numpy as np
from models.actor_critic import LinearPolicy


def multi_loss_test():
    obs_shape = (5,)
    action_shape = (1)
    model = LinearPolicy(obs_shape, action_shape)
    input = torch.randn((1, 5), requires_grad=True)

    def loss(model, input, scalar):
        return (scalar * model(input)).flatten()

    l1 = loss(model, input, torch.tensor([2.], requires_grad=True))
    l2 = loss(model, input, torch.tensor([3.], requires_grad=True))

    l_total = torch.cat((l1, l2))
    l_total.backward(gradient=torch.tensor([1.0, 1.0]))
    params = list(model.parameters())
    grads = [p.grad for p in params]
    pass


if __name__ == '__main__':
    multi_loss_test()
