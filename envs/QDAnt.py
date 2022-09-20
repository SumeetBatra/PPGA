import sys
import gym
import QDgym
import numpy as np
import torch

from signal_slot.signal_slot import *
from QDgym.QDgym_envs import QDAntBulletEnv
from utils.utils import log

# vectorized implementation of QDAnt, using signal-slot model to communicate b/w processes


class QDAntWorker(EventLoopObject):
    def __init__(self, pid, event_loop, object_id, res_buffer, done_buffer, render=False, num_envs=1):
        EventLoopObject.__init__(self, event_loop, object_id)

        self.envs = [gym.make('QDAntBulletEnv-v0', render=render) for _ in range(num_envs)]
        self.pid = pid
        self.res_buffer = res_buffer
        self.done_buffer = done_buffer

    def step(self, action):
        for idx, env in enumerate(self.envs):
            obs, rew, done, info = env.step(action[self.pid])
            obs_tensor, rew_tensor, done_tensor = torch.from_numpy(obs), torch.Tensor([rew]), torch.Tensor([done])
            obs_rew_done = torch.cat((obs_tensor, rew_tensor, done_tensor))
            self.res_buffer[self.pid, idx, :] = obs_rew_done
        self.done_buffer[self.pid] = True

    def reset(self):
        for idx, env in enumerate(self.envs):
            obs = torch.from_numpy(env.reset())
            obs_rew_done = torch.cat((obs, torch.Tensor([0]), torch.Tensor([False])))
            self.res_buffer[self.pid, idx, :] = obs_rew_done
        self.done_buffer[self.pid] = True

    def on_stop(self):
        for env in self.envs:
            env.close()
        self.event_loop.stop()

