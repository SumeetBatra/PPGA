import numpy as np
import torch
import random

from time import time
from envs.vec_env import VecEnv
from utils.utils import log
from RL.ppo import Agent
from utils.vectorized2 import VectorizedPolicy, VectorizedActorCriticShared
from models.actor_critic import ActorCriticSeparate, ActorCriticShared


def test_vec_env():
    num_workers = 4
    envs_per_worker = 1
    vec_env = VecEnv('QDAntBulletEnv-v0', num_workers=num_workers, envs_per_worker=envs_per_worker)
    obs_dim = vec_env.obs_dim
    rand_act = np.random.randn(8)
    vec_env.reset()
    obs, rew, done = vec_env.step(rand_act)
    log.debug(f'{obs=} \n {rew=} \n {done=}')
    log.debug(f'obs shape: {obs.shape}')
    assert obs.shape == torch.Size([num_workers, envs_per_worker, obs_dim + 2])


# def test_throughput():
#     num_workers = 32
#     envs_per_worker = 1
#     vec_env = VecEnv('QDAntBulletEnv-v0', num_workers=num_workers, envs_per_worker=1)
#     vec_env.reset()
#     num_steps = 1000
#     all_obs = []
#     start_time = time()
#     for _ in range(num_steps):
#         rand_act = np.random.randn(8)
#         obs = vec_env.step(rand_act)
#         all_obs.append(obs)
#     elapsed = time() - start_time
#     total_env_steps = num_steps * num_workers * envs_per_worker
#     fps = total_env_steps / elapsed
#     log.debug(f'{fps=}')
#     all_obs = torch.cat(all_obs)
#     log.debug(f"Total obs collected: {all_obs.shape[0]}")


def test_vectorized_inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    obs_shape, action_shape = (8,), np.array(2)
    models = [Agent(obs_shape, action_shape).to(device) for _ in range(10)]
    vec_model = VectorizedPolicy(models, Agent)
    obs = torch.randn((10, 8)).to(device)

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
            k[len("module") + 1 :]: v for k, v in model_state_dict_1.items()
        }

    if next(iter(model_state_dict_2.keys())).startswith("module"):
        model_state_dict_2 = {
            k[len("module") + 1 :]: v for k, v in model_state_dict_2.items()
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
    obs_shape, action_shape = (8,), np.array(2)
    models = [Agent(obs_shape, action_shape) for _ in range(10)]
    vec_model = VectorizedPolicy(models, Agent, obs_shape=obs_shape, action_shape=action_shape)
    models_returned = vec_model.models_list()

    for m_old, m_new in zip(models, models_returned):
        old_statedict, new_statedict = m_old.state_dict(), m_new.state_dict()
        assert validate_state_dicts(old_statedict, new_statedict), "Error: State dicts for original model and model" \
                                                                   " returned by the vectorized model are not the same"

        # double check all parameters are the same
        assert all_params_equal(m_old, m_new), "Error: not all parameters are the same for the original and returned " \
                                               "model"


def test_vectorized_actor_critic_shared_weights():
    device = torch.device('cuda')
    obs_shape, action_shape = (8,), np.array(2)
    models = [ActorCriticShared(obs_shape, action_shape).to(device) for _ in range(10)]
    vec_model = VectorizedActorCriticShared(models, ActorCriticShared, obs_shape=obs_shape, action_shape=action_shape)
    obs = torch.randn((10, 8)).to(device)

    acts_for_loop, vals_for_loop = [], []
    for model, o in zip(models, obs):
        act = model(o)
        val = model.get_value(o)
        acts_for_loop.append(act)
        vals_for_loop.append(val)
    acts_for_loop = torch.cat(acts_for_loop).flatten()
    vals_for_loop = torch.cat(vals_for_loop).flatten()

    acts_vec = vec_model(obs.to(device)).flatten()
    vals_vec = vec_model.get_value(obs.to(device)).flatten()

    assert torch.allclose(acts_vec, acts_for_loop)
    assert torch.allclose(vals_vec, vals_for_loop)


def test_policy_serialize_deserialize():
    obs_shape, action_shape = (8,), np.array(2)
    model1 = ActorCriticShared(obs_shape, action_shape)
    model2 = ActorCriticShared(obs_shape, action_shape)

    params1 = model1.serialize()
    model2.deserialize(params1)

    assert validate_state_dicts(model1.state_dict(), model2.state_dict())
    assert all_params_equal(model1, model2)

# def test_vectorized_stochastic():
#     random.seed(0)
#     np.random.seed(0)
#     torch.manual_seed(0)
#     torch.backends.cudnn.deterministic = True
#
#     torch_state = torch.random.get_rng_state()
#     random_state = random.getstate()
#     numpy_state = np.random.get_state()
#
#     device = torch.device('cuda')
#     obs_shape, action_shape = (8,), np.array(2)
#     models = [Agent(obs_shape, action_shape).to(device) for _ in range(10)]
#     vec_model = VectorizedPolicy(models, Agent).to(device)
#     obs = torch.randn((10, 8)).to(device)
#
#     acts_for_loop, logprobs_for, entropies_for = [], [], []
#     for o, model in zip(obs, models):
#         act, logprob, entropy = model.get_action(o.reshape(1, -1))
#         acts_for_loop.append(act)
#         logprobs_for.append(logprob)
#         entropies_for.append(entropy)
#     acts_for_loop = torch.cat(acts_for_loop).flatten()
#     logprobs_for = torch.cat(logprobs_for).flatten()
#     entropies_for = torch.cat(entropies_for).flatten()
#
#     torch.random.set_rng_state(torch_state)
#     random.setstate(random_state)
#     np.random.set_state(numpy_state)
#     random.seed(0)
#     np.random.seed(0)
#     torch.manual_seed(0)
#
#     acts_vec, logprobs_vec, entropies_vec = vec_model.get_actions(obs, vec_model.action_logstds)
#     acts_vec = acts_vec.flatten()
#
#     assert acts_for_loop == acts_vec, "Error: Actions do not match b/w for-loop and vectorized model inference"
#     assert logprobs_for == logprobs_vec, "Error: Log-probabilities do not match b/w for-loop and vectorized model " \
#                                          "inference"
#     assert entropies_for == entropies_vec, "Error: Entropies do not match b/w for-loop and vectorized model inference"


if __name__ == '__main__':
    test_vec_env()
