import gym
from gym.envs.registration import registry, make, spec


def register(id, *args, **kwargs):
    if id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, *args, **kwargs)


register(id='QDLunarLanderContinuous-v2',
         entry_point='envs.qd_lunar_lander:QDLunarLanderEnv',
         max_episode_steps=1000,
         reward_threshold=500.0)
