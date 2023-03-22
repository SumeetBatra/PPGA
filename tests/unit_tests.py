import numpy as np
import torch
import gym
import random

from attrdict import AttrDict
from time import time
from utils.utilities import log
from models.vectorized import VectorizedPolicy, VectorizedLinearBlock, VectorizedActor
from models.actor_critic import Actor, PGAMEActor
from utils.normalize import ReturnNormalizer, VecRewardNormalizer

TEST_CFG = AttrDict({'normalize_rewards': True, 'normalize_obs': True, 'num_workers': 1, 'envs_per_worker': 1,
            'envs_per_model': 1, 'num_dims': 4, 'gamma': 0.99, 'env_name': 'QDAntBulletEnv-v0', 'seed': 0, 'obs_dim': 28,
                     'obs_shape': (27,), 'action_shape': (8,), 'num_envs': 10})


def test_serialize_deserialize_pgame_actor():
    obs_size, action_shape = 87, (8,)
    agent1 = PGAMEActor(obs_shape=obs_size, action_shape=action_shape)
    agent1_params = agent1.serialize()

    agent2 = PGAMEActor(obs_shape=obs_size, action_shape=action_shape).deserialize(agent1_params)
    agent2_params = agent2.serialize()
    assert np.allclose(agent1_params, agent2_params)


def test_vec_block():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = torch.randn(10, 5, 4).to(device)
    block = VectorizedLinearBlock(weights, device=device)
    obs = torch.randn(20, 4).to(device)

    res_vectorized = block(obs)
    res_for_loop = []

    weights = torch.transpose(weights, 1, 2)
    obs = obs.reshape(10, 2, 4)
    for next_obs, w in zip(obs, weights):
        obs1, obs2 = next_obs[0], next_obs[1]
        res_for_loop.append(obs1 @ w)
        res_for_loop.append(obs2 @ w)
    res_for_loop = torch.cat(res_for_loop).flatten()
    res_vectorized = res_vectorized.flatten()

    assert torch.allclose(res_for_loop, res_vectorized)


# TODO: figure out why this doesn't work
def test_vectorized_policy():
    global TEST_CFG
    dummy_env = make_env(TEST_CFG.env_name, seed=0, gamma=TEST_CFG.gamma)()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    obs_shape, action_shape = dummy_env.observation_space.shape, dummy_env.action_space.shape
    num_models = 10
    models = [Actor(TEST_CFG, obs_shape, action_shape).to(device) for _ in range(num_models)]
    vec_model = VectorizedActor(TEST_CFG, models, Actor).to(device)
    obs = torch.randn((num_models, *obs_shape)).to(device)

    # test same number of models as number of obs
    res_for_loop = []
    start_for = time()
    for o, model in zip(obs, models):
        out = model(o)
        res_for_loop.append(out)
    res_for_loop = torch.cat(res_for_loop)
    elapsed_for = time() - start_for

    start_vec = time()
    res_vectorized = vec_model(obs).flatten()
    elapsed_vec = time() - start_vec

    assert torch.allclose(res_for_loop, res_vectorized), "Error! The vectorized policy does not produce the " \
                                                         "same outputs as naive for-loop over all the individual models"

    print(f'For loop over models took {elapsed_for:.2f} seconds. Vectorized inference took {elapsed_vec:.2f} seconds')

    # test multiple obs per model
    num_models = 7
    num_obs = num_models * 3

    models = [Actor(TEST_CFG, obs_shape, action_shape).to(device) for _ in range(num_models)]
    vec_model = VectorizedActor(TEST_CFG, models, Actor).to(device)
    obs = torch.randn((num_obs, *obs_shape)).to(device)

    with torch.no_grad():
        res_vectorized = vec_model(obs)
        res_for_loop = []
        obs = obs.reshape(num_models, -1, *obs_shape)
        for next_obs, model in zip(obs, models):
            obs1, obs2, obs3 = next_obs[0].reshape(1, -1), next_obs[1].reshape(1, -1), next_obs[2].reshape(1, -1)
            res_for_loop.append(model(obs1))
            res_for_loop.append(model(obs2))
            res_for_loop.append(model(obs3))

    res_for_loop = torch.cat(res_for_loop).flatten()
    res_vectorized = res_vectorized.flatten()

    assert torch.allclose(res_for_loop, res_vectorized)


# from https://gist.github.com/rohan-varma/a0a75e9a0fbe9ccc7420b04bff4a7212
def validate_state_dicts(model_state_dict_1, model_state_dict_2):
    if len(model_state_dict_1) != len(model_state_dict_2):
        log.info(
            f"Length mismatch: {len(model_state_dict_1)}, {len(model_state_dict_2)}"
        )
        return False

    # Replicate modules have "module" attached to their keys, so strip these off when comparing to local model.
    if next(iter(model_state_dict_1.keys())).startswith("module"):
        model_state_dict_1 = {
            k[len("module") + 1:]: v for k, v in model_state_dict_1.items()
        }

    if next(iter(model_state_dict_2.keys())).startswith("module"):
        model_state_dict_2 = {
            k[len("module") + 1:]: v for k, v in model_state_dict_2.items()
        }

    for ((k_1, v_1), (k_2, v_2)) in zip(
            model_state_dict_1.items(), model_state_dict_2.items()
    ):
        if k_1 != k_2:
            log.info(f"Key mismatch: {k_1} vs {k_2}")
            return False
        # convert both to the same CUDA device
        if str(v_1.device) != "cuda:0":
            v_1 = v_1.to("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(v_2.device) != "cuda:0":
            v_2 = v_2.to("cuda:0" if torch.cuda.is_available() else "cpu")

        if not torch.allclose(v_1, v_2):
            log.info(f"Tensor mismatch: {v_1} vs {v_2}")
            return False
    return True


def all_params_equal(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


def test_vectorized_to_list():
    '''Make sure the models_list() function returns the list of models to the exact
    same state they were passed in'''
    obs_shape, action_shape = TEST_CFG.obs_shape, TEST_CFG.action_shape
    models = [Actor(TEST_CFG, obs_shape, action_shape) for _ in range(10)]
    vec_model = VectorizedActor(TEST_CFG, models, Actor, obs_shape=obs_shape, action_shape=action_shape)
    models_returned = vec_model.vec_to_models()

    for m_old, m_new in zip(models, models_returned):
        m_old = m_old.cpu()
        m_new = m_new.cpu()
        old_statedict, new_statedict = m_old.state_dict(), m_new.state_dict()
        assert validate_state_dicts(old_statedict, new_statedict), "Error: State dicts for original model and model" \
                                                                   " returned by the vectorized model are not the same"

        # double check all parameters are the same
        assert all_params_equal(m_old, m_new), "Error: not all parameters are the same for the original and returned " \
                                               "model"


def test_qdvec_to_list():
    obs_shape, action_shape = (8,), np.array(2)
    cfg = {'normalize_rewards': True, 'normalize_obs': True, 'num_workers': 1, 'envs_per_worker': 1,
           'envs_per_model': 1, 'num_dims': 3}
    cfg = AttrDict(cfg)
    models = [Actor(cfg, obs_shape, action_shape, num_dims=3) for _ in range(10)]
    vec_model = VectorizedActor(cfg, models, Actor, measure_dims=3, obs_shape=obs_shape,
                                              action_shape=action_shape)

    vec2models = vec_model.vec_to_models()

    for m_orig, m_new in zip(models, vec2models):
        orig_statedict, new_statedict = m_orig.state_dict(), m_new.state_dict()
        assert validate_state_dicts(orig_statedict, new_statedict), "Error: State dicts for original model and model" \
                                                                    " returned by the vectorized model are not the same"
        # double check all parameters are the same
        assert all_params_equal(m_orig.to(torch.device('cuda')),
                                m_new), "Error: not all parameters are the same for the original and returned " \
                                        "model"


def test_vectorized_rew_normalizer():
    envs_per_model = 5
    num_models = 10
    num_envs = num_models * envs_per_model
    rew_norms = [ReturnNormalizer(envs_per_model) for _ in range(10)]
    means = [norm.return_rms.mean for norm in rew_norms]
    vars = [norm.return_rms.var for norm in rew_norms]
    means = torch.Tensor(means)
    vars = torch.Tensor(vars)
    vec_rew_norm = VecRewardNormalizer(num_envs, num_models, means=means, vars=vars)

    # simulate a rollout
    nrews_for_loop = []
    nrews_vec = []
    for step in range(1000):
        rews = torch.randn(num_envs).reshape(num_models, envs_per_model)
        dones = torch.randint(0, 2, size=(num_envs,)).reshape(num_models, envs_per_model)

        n_for = []
        for rew, done, normalizer in zip(rews, dones, rew_norms):
            rew = normalizer(rew, done)
            n_for.append(rew)
        n_for = torch.cat(n_for)
        nrews_for_loop.append(n_for)

        # do the same for vectorized version
        rews = rews.flatten()
        dones = dones.flatten()
        nrews = vec_rew_norm(rews, dones)
        nrews_vec.append(nrews)

    nrews_for_loop = torch.cat(nrews_for_loop).flatten()
    nrews_vec = torch.cat(nrews_vec).flatten()

    assert torch.allclose(nrews_for_loop, nrews_vec)


if __name__ == '__main__':
    pass
