import sys
import gym
import numpy as np
import torch

from signal_slot.signal_slot import *
from QDgym.QDgym_envs import QDAntBulletEnv
from utils.utils import log
from envs.env import make_env

# vectorized implementation of QDAnt, using signal-slot model to communicate b/w processes


class QDAntWorker(EventLoopObject):
    def __init__(self, cfg, pid, event_loop, object_id, res_buffer, done_buffer, render=False, num_envs=1):
        EventLoopObject.__init__(self, event_loop, object_id)
        self.cfg = cfg
        self.envs = [make_env('QDAntBulletEnv-v0', seed=i, gamma=cfg.gamma)() for i in range(num_envs)]
        self.pid = pid
        self.res_buffer = res_buffer
        self.done_buffer = done_buffer
        self.auto_reset = [False for _ in range(len(self.envs))]

    def step(self, action):
        for idx, env in enumerate(self.envs):
            if self.auto_reset[idx]:
                obs = env.reset()
                rew, done, = 0, False
                self.auto_reset[idx] = False
            else:
                obs, rew, done, info = env.step(action[self.pid])
                if done:
                    self.auto_reset[idx] = True
            obs_tensor, rew_tensor, done_tensor = torch.from_numpy(obs), torch.Tensor([rew]), torch.Tensor([done])
            obs_rew_done = torch.cat((obs_tensor, rew_tensor, done_tensor))
            self.res_buffer[self.pid, idx, :] = obs_rew_done
        self.done_buffer[self.pid] = True

    def reset(self):
        for idx, env in enumerate(self.envs):
            obs  = env.reset()
            obs = torch.from_numpy(obs)
            obs_rew_done = torch.cat((obs, torch.Tensor([0]), torch.Tensor([False])))
            self.res_buffer[self.pid, idx, :] = obs_rew_done
        self.done_buffer[self.pid] = True

    def on_stop(self):
        for env in self.envs:
            env.close()
        self.event_loop.stop()

