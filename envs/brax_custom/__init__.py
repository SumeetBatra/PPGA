# Copyright 2022 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint:disable=g-multiple-import
"""Some example environments to help get started quickly with brax_custom."""

import functools
from typing import Callable, Optional, Type, Union, overload

import brax
from brax.envs import acrobot
from brax.envs import ant
from brax.envs import fast
from brax.envs import fetch
from brax.envs import grasp
from brax.envs import half_cheetah
from brax.envs import hopper
from brax.envs import humanoid
from brax.envs import humanoid_standup
from brax.envs import inverted_double_pendulum
from brax.envs import inverted_pendulum
from brax.envs import pusher
from brax.envs import reacher
from brax.envs import reacherangle
from brax.envs import swimmer
from brax.envs import ur5e
from brax.envs import walker2d
from brax.envs import wrappers
from brax.envs.env import Env, State, Wrapper
import gym

from envs.brax_custom.custom_wrappers.locomotion_wrappers import FeetContactWrapper
from envs.brax_custom.custom_wrappers.reward import TotalReward
from envs.brax_custom.custom_wrappers.clip_wrappers import ActionClipWrapper, RewardClipWrapper, ObservationClipWrapper

# From QDax: experimentally determinated offset (except for antmaze)
# should be sufficient to have only positive rewards but no guarantee
reward_offset = {
    "ant": 3.24,
    "humanoid": 0.0,
    "halfcheetah": 9.231,
    "hopper": 0.9,
    "walker2d": 1.413,
}

_envs = {
    'acrobot': acrobot.Acrobot,
    'ant': functools.partial(ant.Ant, use_contact_forces=True),
    'fast': fast.Fast,
    'fetch': fetch.Fetch,
    'grasp': grasp.Grasp,
    'halfcheetah': half_cheetah.Halfcheetah,
    'hopper': hopper.Hopper,
    'humanoid': humanoid.Humanoid,
    'humanoidstandup': humanoid_standup.HumanoidStandup,
    'inverted_pendulum': inverted_pendulum.InvertedPendulum,
    'inverted_double_pendulum': inverted_double_pendulum.InvertedDoublePendulum,
    'pusher': pusher.Pusher,
    'reacher': reacher.Reacher,
    'reacherangle': reacherangle.ReacherAngle,
    'swimmer': swimmer.Swimmer,
    'ur5e': ur5e.Ur5e,
    'walker2d': walker2d.Walker2d,
}


def get_environment(env_name, **kwargs) -> Env:
    return _envs[env_name](**kwargs)


def register_environment(env_name: str, env_class: Type[Env]):
    _envs[env_name] = env_class


def create(env_name: str,
           episode_length: int = 1000,
           action_repeat: int = 1,
           clip_actions: Optional[tuple] = None,
           clip_rewards: Optional[tuple] = None,
           clip_obs:     Optional[tuple] = None,
           auto_reset: bool = True,
           batch_size: Optional[int] = None,
           eval_metrics: bool = False,
           **kwargs) -> Env:
    """Creates an Env with a specified brax_custom system."""
    env = _envs[env_name](legacy_spring=True, **kwargs)
    env = FeetContactWrapper(env, env_name)
    if clip_obs:
        env = ObservationClipWrapper(env, obs_min=clip_obs[0], obs_max=clip_obs[1])
    if clip_rewards:
        env = RewardClipWrapper(env, rew_min=clip_rewards[0], rew_max=clip_rewards[1])
    if clip_actions:
        env = ActionClipWrapper(env, a_min=clip_actions[0], a_max=clip_actions[1])
    if episode_length is not None:
        env = wrappers.EpisodeWrapper(env, episode_length, action_repeat)
    if batch_size:
        env = wrappers.VectorWrapper(env, batch_size)
    if auto_reset:
        env = wrappers.AutoResetWrapper(env)
    if eval_metrics:
        env = wrappers.EvalWrapper(env)

    return env  # type: ignore


def create_fn(env_name: str, **kwargs) -> Callable[..., Env]:
    """Returns a function that when called, creates an Env."""
    return functools.partial(create, env_name, **kwargs)


@overload
def create_gym_env(env_name: str,
                   batch_size: None = None,
                   seed: int = 0,
                   backend: Optional[str] = None,
                   **kwargs) -> gym.Env:
    ...


@overload
def create_gym_env(env_name: str,
                   batch_size: int,
                   seed: int = 0,
                   backend: Optional[str] = None,
                   **kwargs) -> gym.vector.VectorEnv:
    ...


def create_gym_env(env_name: str,
                   batch_size: Optional[int] = None,
                   seed: int = 0,
                   backend: Optional[str] = None,
                   **kwargs) -> Union[gym.Env, gym.vector.VectorEnv]:
    """Creates a `gym.Env` or `gym.vector.VectorEnv` from a Brax environment."""
    environment = create(env_name=env_name, batch_size=batch_size, **kwargs)
    if batch_size is None:
        return wrappers.GymWrapper(environment, seed=seed, backend=backend)
    if batch_size <= 0:
        raise ValueError(
            '`batch_size` should either be None or a positive integer.')
    return wrappers.VectorGymWrapper(environment, seed=seed, backend=backend)
