import gym
import numpy as np

from gym.envs.box2d import LunarLanderContinuous


class QDLunarLanderEnv(LunarLanderContinuous):
    def __init__(self):
        self.T = 0
        self.tot_reward = 0.0
        self.all_y_vels = []
        self.impact_x_pos = None
        self.impact_y_vel = None
        super().__init__()

        print(f"The behavioural desciptor is 2-dimentional",
              f"and defined as the the x impact-location and y-velocity on impact")

    def reset(self):
        r = super().reset()
        self.T = 0
        self.tot_reward = 0.0
        self.all_y_vels = []
        self.impact_x_pos = None
        self.impact_y_vel = None

        return r

    def step(self, action):
        obs, reward, done, info = super().step(action)
        # Refer to the definition of state here:
        # https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py#L306
        x_pos = obs[0]
        y_vel = obs[3]
        leg0_touch = bool(obs[6])
        leg1_touch = bool(obs[7])
        self.all_y_vels.append(y_vel)

        # Check if the lunar lander is impacting for the first time.
        if self.impact_x_pos is None and (leg0_touch or leg1_touch):
            self.impact_x_pos = x_pos
            self.impact_y_vel = y_vel
        info['desc'] = [self.impact_x_pos, self.impact_y_vel]
        info['x_pos'] = x_pos
        info['y_vel'] = y_vel
        return obs, reward, done, info



