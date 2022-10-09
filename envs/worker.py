import torch
import numpy as np

from signal_slot.signal_slot import *
from envs.env import make_env


class Worker(EventLoopObject):
    def __init__(self, cfg, pid, event_loop, object_id, res_buffer, done_buffer, infos, render=False, num_envs=1):
        EventLoopObject.__init__(self, event_loop, object_id)
        self.cfg = cfg
        seeds = np.random.random_integers(0, 10000, size=num_envs)
        # seed = 0  # make the environment deterministic for debugging
        self.envs = [make_env(cfg.env_name, seed=seeds[i].item(), gamma=cfg.gamma)() for i in range(num_envs)]
        self.pid = pid
        self.res_buffer = res_buffer
        self.done_buffer = done_buffer
        self.infos = infos

    def step(self, action, autoreset):
        action = action.reshape(self.cfg.num_workers, self.cfg.envs_per_worker, -1)
        for idx, env in enumerate(self.envs):
            obs, rew, done, info = env.step(action[self.pid, idx])
            if done:
                self.infos['total_reward'][self.pid, idx] = env.total_reward
                self.infos['traj_length'][self.pid, idx] = env.T
                # the actual behavior descriptor that's dependent on the entire trajectory
                bc = torch.from_numpy(info['bc'])
                self.infos['bc'][self.pid, idx, :] = bc
                if autoreset:
                    obs = env.reset()
                else:
                    rew = 0
            obs_tensor, rew_tensor, done_tensor = torch.from_numpy(obs), torch.Tensor([rew]), torch.Tensor([done])
            measures = torch.from_numpy(info['measures'])
            res = torch.cat((obs_tensor, rew_tensor, done_tensor))
            self.res_buffer[self.pid, idx, :] = res
            self.infos['measures'][self.pid, idx, :] = measures
        self.done_buffer[self.pid] = True

    def reset(self):
        for idx, env in enumerate(self.envs):
            obs = env.reset()
            obs = torch.from_numpy(obs)
            measures = torch.zeros(4)
            res = torch.cat((obs, torch.Tensor([0]), torch.Tensor([False])))
            self.res_buffer[self.pid, idx, :] = res
            self.infos['total_reward'][:] = 0
            self.infos['bc'][:] = measures
        self.done_buffer[self.pid] = True

    def on_stop(self):
        for env in self.envs:
            env.close()
        self.event_loop.stop()
