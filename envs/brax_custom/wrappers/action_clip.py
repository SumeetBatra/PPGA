from brax.envs import env as brax_env
from brax import jumpy as jp


class ActionClipWrapper(brax_env.Wrapper):
    def __init__(self, env: brax_env.Env, a_min, a_max):
        super().__init__(env)
        self.a_min = a_min
        self.a_max = a_max

    def reset(self, rng: jp.ndarray) -> brax_env.State:
        state = self.env.reset(rng)
        return state

    def step(self, state: brax_env.State, action: jp.ndarray) -> brax_env.State:
        action = jp.clip(action, self.a_min, self.a_max)
        nstate = self.env.step(state, action)
        return nstate