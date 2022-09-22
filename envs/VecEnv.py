import torch
import gym
import numpy as np
from signal_slot.signal_slot import *

import envs.QDAnt
from utils.utils import log
from envs.env import make_env


env_dispatch = {
    'QDAntBulletEnv-v0': envs.QDAnt.QDAntWorker
}


class VecEnv(EventLoopObject, gym.Env):
    def __init__(self, cfg, env_name: str, num_workers, envs_per_worker=1):
        self.cfg = cfg
        process = EventLoopProcess('main')
        EventLoopObject.__init__(self, process.event_loop, 'QDVecEnv')
        gym.Env.__init__(self)

        dummy_env = make_env(env_name, seed=0, gamma=cfg.gamma)()
        self.single_observation_space = dummy_env.observation_space
        self.single_action_space = dummy_env.action_space
        self.obs_dim = dummy_env.observation_space.shape[0]
        self.obs_shape = dummy_env.observation_space.shape
        self.action_space = dummy_env.action_space
        dummy_env.close()

        # each worker will return obs + scalar reward + scalar done flag concatenated together
        self.res_buffer = torch.zeros((num_workers, envs_per_worker, self.obs_dim + 2)).share_memory_()
        self.done_buffer = torch.zeros((num_workers,)).share_memory_()
        self.worker_processes = [EventLoopProcess(f'process_{i}') for i in range(num_workers)]
        self.workers = [env_dispatch[env_name](cfg,
                                               i,
                                               self.worker_processes[i].event_loop,
                                               f'worker_{self.worker_processes[i].object_id}',
                                               self.res_buffer,
                                               self.done_buffer,
                                               render=False,
                                               num_envs=envs_per_worker) for i in range(num_workers)]
        self.num_envs = num_workers * envs_per_worker
        self.connect_signals_to_slots()
        for proc in self.worker_processes:
            proc.start()

    @signal
    def step_signal(self):
        pass

    @signal
    def reset_signal(self):
        pass

    @signal
    def stop(self):
        pass

    def step(self, action):
        self.step_signal.emit(action)
        while not all(self.done_buffer):
            ...
        obs, rew, done = self.res_buffer[:, :, :self.obs_dim], self.res_buffer[:, :, -2], self.res_buffer[:, :, -1]
        return obs.reshape(self.num_envs, -1), rew.reshape(self.num_envs, -1), done.reshape(self.num_envs, -1), {}

    def reset(self):
        self.reset_signal.emit()
        while not all(self.done_buffer):
            ...
        obs, rew, done = self.res_buffer[:, :, :self.obs_dim], self.res_buffer[:, :, -2], self.res_buffer[:, :, -1]
        return obs.reshape(self.num_envs, -1)

    def connect_signals_to_slots(self):
        for worker in self.workers:
            self.step_signal.connect(worker.step)
            self.reset_signal.connect(worker.reset)
            self.stop.connect(worker.on_stop)
