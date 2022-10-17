import copy

import torch
import gym
from signal_slot.signal_slot import *

from envs.cpu.env import make_env
from envs.cpu.worker import Worker


def make_vec_env(cfg):
    vec_env = VecEnv(cfg,
                     cfg.env_name,
                     num_workers=cfg.num_workers,
                     envs_per_worker=cfg.envs_per_worker,
                     seed=cfg.seed)
    return vec_env


def make_vec_env_for_eval(cfg, num_workers, envs_per_worker):
    vec_env = VecEnv(cfg,
                     cfg.env_name,
                     num_workers,
                     envs_per_worker)

    return vec_env


class VecEnv(EventLoopObject, gym.Env):
    def __init__(self, cfg, env_name: str, num_workers, envs_per_worker=1, double_buffered_sampling=False, seed=None):
        self.cfg = cfg
        process = EventLoopProcess('main')
        EventLoopObject.__init__(self, process.event_loop, 'QDVecEnv')
        gym.Env.__init__(self)
        self.double_buffered_sampling = double_buffered_sampling
        self.num_workers = num_workers
        self.envs_per_worker = envs_per_worker
        self.seed = seed  # if the seed is set, then all envs will get the same seed, making the environments deterministic. Good for debugging

        dummy_env = make_env(env_name, seed=0, gamma=cfg.gamma)()
        self.single_observation_space = dummy_env.observation_space
        self.single_action_space = dummy_env.action_space
        self.obs_dim = dummy_env.observation_space.shape[0]
        self.obs_shape = dummy_env.observation_space.shape
        self.action_space = dummy_env.action_space
        dummy_env.close()

        # each worker will return obs + scalar reward + scalar done flag + n-dim measures concatenated together
        self.measure_dim = cfg.num_dims
        reward_dim = 1
        done_dim = 1
        total_dims = self.obs_dim + reward_dim + done_dim
        self.res_buffer = torch.zeros((num_workers, envs_per_worker, total_dims)).share_memory_()
        self.done_buffer = torch.zeros((num_workers,)).share_memory_()
        self.infos = {'total_reward': torch.zeros(num_workers, envs_per_worker).share_memory_(),
                      'traj_length': torch.zeros(num_workers, envs_per_worker).share_memory_(),
                      # final measures averaged over length of trajectory (non-markovian)
                      'bc': torch.zeros((num_workers, envs_per_worker, self.measure_dim)).share_memory_(),
                      # per timestep measure (markovian??)
                      'measures': torch.zeros((num_workers, envs_per_worker, self.measure_dim)).share_memory_()}
        self.worker_processes = [EventLoopProcess(f'process_{i}') for i in range(num_workers)]
        self.workers = [Worker(cfg,
                               i,
                               self.worker_processes[i].event_loop,
                               f'worker_{self.worker_processes[i].object_id}',
                               self.res_buffer,
                               self.done_buffer,
                               self.infos,
                               render=False,
                               num_envs=envs_per_worker,
                               seed=self.seed) for i in range(num_workers)]
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

    def step(self, action, autoreset=True):
        self.step_signal.emit(action, autoreset)
        while not all(self.done_buffer):
            ...
        res = self.res_buffer.detach().clone().reshape(self.num_envs, -1)
        obs, rew, done = res[:, :self.obs_dim], res[:, self.obs_dim], res[:, self.obs_dim + 1]
        infos = copy.deepcopy(self.infos)
        for key, val in infos.items():
            infos[key] = val.reshape(self.num_envs, -1)
        # reset the done buffers
        self.done_buffer[:] = 0
        return obs, rew, done, infos

    def reset(self):
        self.reset_signal.emit()
        while not all(self.done_buffer):
            ...
        res = self.res_buffer.detach().clone().reshape(self.num_envs, -1)
        obs = res[:, :self.obs_dim]
        self.done_buffer[:] = 0
        return obs

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
