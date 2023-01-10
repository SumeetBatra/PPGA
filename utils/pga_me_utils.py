import jax
import jax.numpy as jnp
import numpy as np
import functools
import os

import torch
from qdax.core.neuroevolution.networks.networks import MLP
from qdax import environments
from qdax.types import Genotype
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from jax._src.flatten_util import ravel_pytree
from utils.archive_utils import load_pga_me_archive

from models.actor_critic import PGAMEActor
from ribs.archives import GridArchive


def pgame_repertoire_to_pyribs_archive(cp_path):
    # define the environment
    env_name = 'walker2d_uni'
    seed = 1111
    episode_length = 1000
    env = environments.create(env_name, episode_length=episode_length)
    env_batch_size = 1

    # define the MLP architecture
    policy_layer_sizes = (128, 128) + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    # setup initial variables and rng keys
    random_key = jax.random.PRNGKey(seed)
    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=env_batch_size)
    fake_batch = jnp.zeros(shape=(env_batch_size, env.observation_size))
    init_variables = jax.vmap(policy_network.init)(keys, fake_batch)

    def load_archive(random_key):
        random_key, subkey = jax.random.split(random_key)
        fake_batch = jnp.zeros(shape=(env.observation_size,))
        fake_params = policy_network.init(subkey, fake_batch)

        _, reconstruction_fn = ravel_pytree(fake_params)
        repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=cp_path)
        return repertoire

    repertoire = load_archive(random_key)
    active_inds = jnp.where(repertoire.fitnesses != -jnp.inf)[0]

    flax_params = repertoire.genotypes['params']

    def flax_to_torch_model(model_ind):
        pytorch_model = PGAMEActor(obs_shape=env.observation_size, action_shape=(env.action_size,))
        pytorch_params = dict(pytorch_model.named_parameters())
        for i in range(len(flax_params)):
            pytorch_params[f'actor_mean.{2*i}.weight'].data = torch.from_numpy(flax_params[f'Dense_{i}']['kernel'][model_idx]._value.T)
            pytorch_params[f'actor_mean.{2*i}.bias'].data = torch.from_numpy(flax_params[f'Dense_{i}']['bias'][model_idx]._value.T)
        return pytorch_model.serialize()

    solution_batch = []
    for model_idx in active_inds:
        solution_batch.append(flax_to_torch_model(model_idx))
        print(f'Finished process model {model_idx}')
    solution_batch = np.array(solution_batch)
    obj_batch = np.array(repertoire.fitnesses[active_inds])
    measures_batch = np.array(repertoire.descriptors[active_inds])

    archive_dims = [100, 100]
    num_dims = 2
    seed = 1111
    bounds = [(0., 1.0) for _ in range(num_dims)]
    archive = GridArchive(solution_dim=solution_batch.shape[1],
                          dims=archive_dims,
                          ranges=bounds,
                          threshold_min=-np.inf,
                          seed=seed)
    archive.add(solution_batch, obj_batch, measures_batch)
    return archive



if __name__ == '__main__':
    # checkpoint_dir = '/home/sumeet/QDax/experiments/pga_me_ant_uni_testrun_seed_1111/checkpoints/checkpoint_00399'
    # archive = load_pga_me_archive(checkpoint_dir)
    # archive_df = archive.as_pandas()
    # elites = archive_df.query("objective > 5000").sort_values("objective", ascending=False)
    # pass

    cp_path = '/home/sumeet/QDax/experiments/walker2d_checkpoint/checkpoint_00731/'
    archive = pgame_repertoire_to_pyribs_archive(cp_path)
    archive_df = archive.as_pandas(include_solutions=True)
    archive_df.to_pickle(cp_path + 'ribs_archive.pkl')
