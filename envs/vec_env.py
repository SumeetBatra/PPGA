import torch
import gym
import numpy as np
from signal_slot.signal_slot import *

from utils.utils import log
from envs.env import make_env
from envs.worker import Worker


class VecEnv(EventLoopObject, gym.Env):
    def __init__(self, cfg, env_name: str, num_workers, envs_per_worker=1, double_buffered_sampling=False):
        self.cfg = cfg
        process = EventLoopProcess('main')
        EventLoopObject.__init__(self, process.event_loop, 'QDVecEnv')
        gym.Env.__init__(self)
        self.double_buffered_sampling = double_buffered_sampling
        self.num_workers = num_workers

        dummy_env = make_env(env_name, seed=0, gamma=cfg.gamma)()
        self.single_observation_space = dummy_env.observation_space
        self.single_action_space = dummy_env.action_space
        self.obs_dim = dummy_env.observation_space.shape[0]
        self.obs_shape = dummy_env.observation_space.shape
        self.action_space = dummy_env.action_space
        dummy_env.close()

        # each worker will return obs + scalar reward + scalar done flag concatenated together
        # TODO: make this general
        measure_dim = 2
        reward_dim = 1
        done_dim = 1
        total_dims = self.obs_dim + reward_dim + done_dim + measure_dim
        self.res_buffer = torch.zeros((num_workers, envs_per_worker, total_dims)).share_memory_()
        self.done_buffer = torch.zeros((num_workers,)).share_memory_()
        self.worker_processes = [EventLoopProcess(f'process_{i}') for i in range(num_workers)]
        self.workers = [Worker(cfg,
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
    def step_buffer1(self):
        pass

    @signal
    def step_buffer2(self):
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
        obs, rew, done = self.res_buffer[:, :, :self.obs_dim], self.res_buffer[:, :, self.obs_dim + 1], \
                         self.res_buffer[:, :, self.obs_dim + 2]
        infos = {'desc': self.res_buffer[:, :, -2:]}
        return obs.reshape(self.num_envs, -1), rew.reshape(self.num_envs, -1), done.reshape(self.num_envs, -1), infos

    def reset(self):
        self.reset_signal.emit()
        while not all(self.done_buffer):
            ...
        obs, rew, done = self.res_buffer[:, :, :self.obs_dim], self.res_buffer[:, :, -2], self.res_buffer[:, :, -1]
        return obs.reshape(self.num_envs, -1)

    def connect_signals_to_slots(self):
        if self.double_buffered_sampling:
            group1, group2 = self.workers[:self.num_workers // 2], self.workers[self.num_workers // 2:]
            for worker in group1:
                self.step_buffer1.connect(worker.step)
                self.reset_signal.connect(worker.reset)
                self.stop.connect(worker.on_stop)
            for worker in group2:
                self.step_buffer2.connect(worker.step)
                self.reset_signal.connect(worker.reset)
                self.stop.connect(worker.on_stop)
        else:
            for worker in self.workers:
                self.step_signal.connect(worker.step)
                self.reset_signal.connect(worker.reset)
                self.stop.connect(worker.on_stop)
