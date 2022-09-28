import torch

from signal_slot.signal_slot import *
from envs.env import make_env


class Worker(EventLoopObject):
    def __init__(self, cfg, pid, event_loop, object_id, res_buffer, done_buffer, render=False, num_envs=1):
        EventLoopObject.__init__(self, event_loop, object_id)
        self.cfg = cfg
        self.envs = [make_env(cfg.env_name, seed=i, gamma=cfg.gamma)() for i in range(num_envs)]
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
            if info['desc'][0] is None:
                measures = torch.BoolTensor([False, False])  # TODO: make this general
            else:
                measures = torch.from_numpy(info['desc'])
            obs_rew_done_measure = torch.cat((obs_tensor, rew_tensor, done_tensor, measures))
            self.res_buffer[self.pid, idx, :] = obs_rew_done_measure
        self.done_buffer[self.pid] = True

    def reset(self):
        for idx, env in enumerate(self.envs):
            obs = env.reset()
            obs = torch.from_numpy(obs)
            obs_rew_done_measures = torch.cat((obs, torch.Tensor([0]), torch.Tensor([False]), torch.Tensor([0, 0])))
            self.res_buffer[self.pid, idx, :] = obs_rew_done_measures
        self.done_buffer[self.pid] = True

    def on_stop(self):
        for env in self.envs:
            env.close()
        self.event_loop.stop()
