import numpy as np
import random
import copy

import torch
import time
import torch.nn as nn
import wandb
from collections import deque
from envs.cpu.vec_env import make_vec_env, make_vec_env_for_eval
from envs.brax_custom.gpu_env import make_vec_env_brax
from utils.utils import log, save_checkpoint
from models.vectorized import VectorizedActor
from models.actor_critic import Actor, Critic, QDCritic


# based off of the clean-rl implementation
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py


class PPO:
    def __init__(self, seed, cfg, vec_env):
        self.vec_env = vec_env
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.obs_shape = self.vec_env.single_observation_space.shape
        self.action_shape = self.vec_env.single_action_space.shape

        agent = Actor(cfg, self.obs_shape, self.action_shape).to(self.device)
        self._agents = [agent]
        critic = QDCritic(self.obs_shape, measure_dim=cfg.num_dims).to(self.device)
        self._critic = critic
        self.vec_inference = VectorizedActor(cfg, self._agents, Actor, obs_shape=self.obs_shape,
                                             action_shape=self.action_shape).to(self.device)
        self.vec_optimizer = torch.optim.Adam(self.vec_inference.parameters(), lr=cfg.learning_rate, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg.learning_rate, eps=1e-5)
        self.cfg = cfg

        # # single policy eval env
        # single_eval_cfg = copy.deepcopy(cfg)
        # single_eval_cfg.num_workers = 4
        # single_eval_cfg.envs_per_worker = 1
        # single_eval_cfg.envs_per_model = 4
        # self.single_eval_env = make_vec_env_for_eval(single_eval_cfg, num_workers=single_eval_cfg.num_workers,
        #                                              envs_per_worker=single_eval_cfg.envs_per_worker)
        #

        # multi-policy eval
        if cfg.env_type == 'cpu':
            multi_eval_cfg = copy.deepcopy(cfg)
            multi_eval_cfg.num_workers = cfg.mega_lambda
            multi_eval_cfg.envs_per_worker = 4
            multi_eval_cfg.envs_per_model = 4
            self.multi_eval_env = make_vec_env_for_eval(multi_eval_cfg, num_workers=multi_eval_cfg.num_workers,
                                                        envs_per_worker=multi_eval_cfg.envs_per_worker)
        else:
            # brax env
            multi_eval_cfg = copy.deepcopy(cfg)
            multi_eval_cfg.env_batch_size = 1024
            self.multi_eval_env = make_vec_env_brax(cfg)

            # metrics for logging
        self.metric_last_n_window = 10
        self.episodic_returns = deque([], maxlen=self.metric_last_n_window)
        self.episodic_returns.append(0)
        self._report_interval = cfg.report_interval
        self.num_intervals = 0
        self.total_rewards = torch.zeros(self.vec_env.num_envs)

        # seeding
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = cfg.torch_deterministic

        # initialize tensors for training
        self.obs = torch.zeros(
            (cfg.rollout_length, self.vec_env.num_envs) + self.vec_env.single_observation_space.shape).to(
            self.device)
        self.actions = torch.zeros(
            (cfg.rollout_length, self.vec_env.num_envs) + self.vec_env.single_action_space.shape).to(
            self.device)
        self.logprobs = torch.zeros((cfg.rollout_length, self.vec_env.num_envs)).to(self.device)
        self.rewards = torch.zeros((cfg.rollout_length, self.vec_env.num_envs)).to(self.device)
        self.dones = torch.zeros((cfg.rollout_length, self.vec_env.num_envs)).to(self.device)
        self.values = torch.zeros((cfg.rollout_length, self.vec_env.num_envs)).to(self.device)
        self.measures = torch.zeros((cfg.rollout_length, self.vec_env.num_envs, self.cfg.num_dims)).to(self.device)
        self.measure_values = torch.zeros_like(self.measures).to(self.device)

        next_obs = self.vec_env.reset()
        self.next_obs = next_obs.to(self.device)
        self.next_done = torch.zeros(self.vec_env.num_envs).to(self.device)

    @property
    def agents(self):
        return self.vec_inference.vec_to_models()

    @agents.setter
    def agents(self, agents):
        self._agents = agents
        self.vec_inference = VectorizedActor(self.cfg, self._agents, Actor,
                                             obs_shape=self.obs_shape,
                                             action_shape=self.action_shape)
        self.vec_optimizer = torch.optim.Adam(self.vec_inference.parameters(), lr=self.cfg.learning_rate, eps=1e-5)

    def calculate_rewards(self, next_obs, next_done, rewards, values, dones, rollout_length, dqd=False):
        # bootstrap value if not done
        with torch.no_grad():
            if dqd:
                next_obj_val = self._critic.get_value(next_obs)
                next_m_vals = self._critic.get_measure_values(next_obs).reshape(self.cfg.num_envs, -1)
                next_value = torch.cat((next_obj_val, next_m_vals), dim=1)
                mask = torch.eye(self.cfg.num_dims + 1).to(self.device)
                envs_per_dim = self.cfg.num_envs // (self.cfg.num_dims + 1)
                mask = torch.repeat_interleave(mask, dim=0, repeats=envs_per_dim)
                next_value = (next_value * mask).sum(dim=1).reshape(1, -1)
            else:
                next_value = self._critic.get_value(next_obs).reshape(1, -1).to(self.device)
            # assume we use gae
            advantages = torch.zeros_like(rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(rollout_length)):
                if t == rollout_length - 1:
                    is_next_nonterminal = 1.0 - next_done.long()
                    next_values = next_value
                else:
                    is_next_nonterminal = 1.0 - dones[t + 1]
                    next_values = values[t + 1]
                if dqd:
                    is_next_nonterminal = is_next_nonterminal.to(self.device)
                else:
                    is_next_nonterminal = is_next_nonterminal.to(self.device).reshape(1, -1)
                delta = rewards[t] + self.cfg.gamma * next_values * is_next_nonterminal - values[t]
                advantages[t] = lastgaelam = \
                    delta + self.cfg.gamma * self.cfg.gae_lambda * is_next_nonterminal * lastgaelam
            returns = advantages + values
        return advantages, returns

    def batch_update(self, values, batched_data, dqd=False):
        b_values = values
        (b_obs, b_logprobs, b_actions, b_advantages, b_returns) = batched_data
        batch_size = b_obs.shape[1]
        minibatch_size = batch_size // self.cfg.num_minibatches

        obs_dim, action_dim = self.obs_shape[0], self.action_shape[0]

        b_inds = torch.arange(batch_size)
        clipfracs = []
        for epoch in range(self.cfg.update_epochs):
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy = self.vec_inference.get_action(b_obs[:, mb_inds].reshape(-1, obs_dim),
                                                                       b_actions[:, mb_inds].reshape(-1, action_dim))

                if dqd:
                    next_obj_val = self._critic.get_value(b_obs[:, mb_inds])
                    next_m_vals = self._critic.get_measure_values(b_obs[:, mb_inds])
                    next_value = torch.cat((next_obj_val, next_m_vals), dim=2)
                    mask = torch.eye(self.cfg.num_dims + 1).unsqueeze(dim=1).to(self.device)
                    newvalue = (next_value * mask).sum(dim=2)
                else:
                    newvalue = self._critic.get_value(b_obs[:, mb_inds])

                logratio = newlogprob - b_logprobs[:, mb_inds].flatten()
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.cfg.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[:, mb_inds].flatten()
                if self.cfg.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.cfg.clip_coef, 1 + self.cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # value loss
                newvalue = newvalue.view(-1)
                if self.cfg.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[:, mb_inds].flatten()) ** 2
                    v_clipped = b_values[:, mb_inds].flatten() + torch.clamp(
                        newvalue - b_values[:, mb_inds].flatten(),
                        -self.cfg.clip_coef,
                        self.cfg.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[:, mb_inds].flatten()) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[:, mb_inds].flatten()) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.cfg.entropy_coef * entropy_loss + v_loss * self.cfg.vf_coef

                self.vec_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.vec_inference.parameters(), self.cfg.max_grad_norm)
                self.vec_optimizer.step()
                nn.utils.clip_grad_norm_(self._critic.parameters(), self.cfg.max_grad_norm)
                self.critic_optimizer.step()

            if self.cfg.target_kl is not None:
                if approx_kl > self.cfg.target_kl:
                    break

        return pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs

    def get_measure_grads(self, num_updates, rollout_length):
        global_step = 0
        original_params = copy.deepcopy(self._agents[0].serialize())
        agent_params = [copy.deepcopy(self._agents[0]).serialize() for _ in range(self.cfg.num_dims)]
        agents = [Actor(self.cfg, self.obs_shape, self.action_shape).deserialize(agent_params[i]).to(self.device) for i
                  in range(self.cfg.num_dims)]
        num_agents = len(agents)  # this is essentially the number of measures
        self.agents = agents  # this also updates the optimizers and vec_inference

        start = time.time()
        for update in range(1, num_updates + 1):
            for step in range(rollout_length):
                global_step += self.vec_env.num_envs
                if self.cfg.normalize_obs:
                    self.next_obs = self.vec_inference.vec_normalize_obs(self.next_obs)
                self.obs[step] = self.next_obs
                self.dones[step] = self.next_done.view(-1)

                with torch.no_grad():
                    action, logprob, _ = self.vec_inference.get_action(self.next_obs)
                    obs_per_measure = self.next_obs.reshape(num_agents, self.vec_env.num_envs // num_agents, -1)
                    measure_vals = []
                    # TODO: can this be vectorized?
                    for i, obs in enumerate(obs_per_measure):
                        measure_value = self._critic.get_measure_value(obs, dim=i).flatten()
                        measure_vals.append(measure_value)
                    measure_vals = torch.cat(measure_vals)
                    self.measure_values[step] = measure_vals

                self.actions[step] = action
                self.logprobs[step] = logprob

                self.next_obs, _, self.next_done, infos = self.vec_env.step(action.cpu().numpy())
                measures = infos['measures']
                # if self.cfg.normalize_rewards:  # TODO: make this a separate flag
                #     measures = self.vec_inference.vec_normalize_measures(measures.cpu(), self.next_done)
                self.measures[step] = measures

            # finished the rollout, now we will calculate advantages for the measures
            advantages, returns = self.calculate_rewards(self.next_obs, self.next_done, self.measures,
                                                         self.measure_values, self.dones,
                                                         rollout_length=self.cfg.rollout_length, measure_reward=True)

            # use the data to get m_grads and update the respective policies

            # flatten the batch
            b_obs = self.obs.reshape((num_agents, -1,) + self.vec_env.single_observation_space.shape)
            b_logprobs = self.logprobs.reshape(num_agents, -1)
            b_actions = self.actions.reshape((num_agents, -1,) + self.vec_env.single_action_space.shape)
            b_advantages = advantages.reshape(num_agents, -1)
            b_returns = returns.reshape(num_agents, -1)
            b_values = self.values.reshape(num_agents, -1)

            for i in range(self.vec_inference.num_models):
                (pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs) = self.update(
                    b_values[i], (b_obs[i], b_logprobs[i], b_actions[i], b_advantages[i], b_returns[i]),
                    self.actor_optimizers[i], i, measures=True)

            # update vec inference
            # TODO: create an update_params() method instead of recreating vec_inf each time
            self.vec_inference = VectorizedActor(self.cfg, self._agents, Actor,
                                                 obs_shape=self.obs_shape,
                                                 action_shape=self.action_shape)

            if self.cfg.use_wandb:
                wandb.log({
                    "QDLosses/measures_value_loss": v_loss.item(),
                    "QDLosses/measures_policy_loss": pg_loss.item(),
                    "QDLosses/measures_old_approx_kl": old_approx_kl.item(),
                    "QDLosses/measures_approx_kl": approx_kl.item(),
                    "QDLosses/measures_clipfrac": np.mean(clipfracs)
                })

            log.debug(f'Finished PPO iteration {update} for measures')

        m_grads = np.array([self._agents[i].serialize() - original_params for i in range(num_agents)])
        m_grads = m_grads.reshape(1, self.cfg.num_dims, -1)  # TODO: fix this
        elapsed = time.time() - start
        log.debug(f'Calculating measure grads took {elapsed:.2f} seconds to complete')

        return m_grads

    def train(self, num_updates, rollout_length, dqd=False):
        global_step = 0

        if dqd:
            # TODO: make this work for multiple emitters
            solution_params = self._agents[0].serialize()
            # create copy of agent for f and one of each m
            agent_original_params = [copy.deepcopy(solution_params) for _ in range(self.cfg.num_dims + 1)]
            agents = [Actor(self.cfg, self.obs_shape, self.action_shape).deserialize(params) for params in
                      agent_original_params]
            self.agents = agents

        num_agents = len(self._agents)

        train_start = time.time()
        for update in range(1, num_updates + 1):
            for step in range(rollout_length):
                global_step += self.vec_env.num_envs
                if self.cfg.normalize_obs:
                    self.next_obs = self.vec_inference.vec_normalize_obs(self.next_obs)
                self.obs[step] = self.next_obs
                self.dones[step] = self.next_done.view(-1)

                with torch.no_grad():
                    action, logprob, _ = self.vec_inference.get_action(self.next_obs)
                    value = self._critic.get_value(self.next_obs)
                    if dqd:
                        measure_values = self._critic.get_measure_values(self.next_obs)
                        self.measure_values[step] = measure_values
                    self.values[step] = value.flatten()
                self.actions[step] = action
                self.logprobs[step] = logprob

                self.next_obs, reward, self.next_done, infos = self.vec_env.step(action.cpu().numpy())
                measures = infos['measures']
                self.measures[step] = measures
                reward = reward.cpu()
                self.total_rewards += reward
                self.next_obs = self.next_obs.to(self.device)
                if self.cfg.normalize_rewards:
                    reward = self.vec_inference.vec_normalize_rewards(reward, self.next_done)
                self.rewards[step] = reward.squeeze()

                # TODO: move this to a separate process
                if self.num_intervals % self._report_interval == 0:
                    for i, done in enumerate(self.next_done.flatten()):
                        if done:
                            total_reward = self.total_rewards[i].clone()
                            # log.debug(f'{total_reward=}')
                            self.episodic_returns.append(total_reward)
                            self.total_rewards[i] = 0
                self.num_intervals += 1

            if dqd:
                # concat obj and measure values and mask them appropriately
                obj_measure_values = torch.cat((self.values.unsqueeze(dim=2), self.measure_values), dim=2)
                envs_per_dim = self.cfg.num_envs // (self.cfg.num_dims + 1)
                mask = torch.eye(self.cfg.num_dims + 1)
                mask = torch.repeat_interleave(mask, dim=0, repeats=envs_per_dim).unsqueeze(dim=0).to(self.device)
                obj_measure_values = (obj_measure_values * mask).sum(dim=2)

                # concat the reward w/ measures and mask appropriately
                rew_measures = torch.cat((self.rewards.unsqueeze(dim=2), self.measures), dim=2)
                rew_measures = (rew_measures * mask).sum(dim=2)
                advantages, returns = self.calculate_rewards(self.next_obs, self.next_done, rew_measures,
                                                             obj_measure_values, self.dones,
                                                             rollout_length=self.cfg.rollout_length, dqd=True)
            else:
                advantages, returns = self.calculate_rewards(self.next_obs, self.next_done, self.rewards, self.values,
                                                             self.dones,
                                                             rollout_length=self.cfg.rollout_length)
            # flatten the batch
            b_obs = self.obs.transpose(0, 1).reshape((num_agents, -1,) + self.vec_env.single_observation_space.shape)
            b_logprobs = self.logprobs.transpose(0, 1).reshape(num_agents, -1)
            b_actions = self.actions.transpose(0, 1).reshape((num_agents, -1,) + self.vec_env.single_action_space.shape)
            b_advantages = advantages.transpose(0, 1).reshape(num_agents, -1)
            b_returns = returns.transpose(0, 1).reshape(num_agents, -1)
            b_values = self.values.transpose(0, 1).reshape(num_agents, -1)

            # update the network
            (pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs) = self.batch_update(b_values,
                                                                                                     (b_obs,
                                                                                                      b_logprobs,
                                                                                                      b_actions,
                                                                                                      b_advantages,
                                                                                                      b_returns),
                                                                                                      dqd=dqd)

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            avg_log_stddev = self.vec_inference.actor_logstd.mean().detach().cpu().numpy()
            avg_obj_magnitude = self.rewards.mean()

            if self.cfg.use_wandb:
                wandb.log({
                    "charts/actor_avg_logstd": avg_log_stddev,
                    "charts/average_rew_magnitude": avg_obj_magnitude,
                    "losses/value_loss": v_loss.item(),
                    "losses/policy_loss": pg_loss.item(),
                    "losses/entropy": entropy_loss.item(),
                    "losses/old_approx_kl": old_approx_kl.item(),
                    "losses/approx_kl": approx_kl.item(),
                    "losses/clipfrac": np.mean(clipfracs),
                    "losses/explained_variance": explained_var,
                    f"Average Episodic Reward": sum(self.episodic_returns) / len(self.episodic_returns),
                    "Env step": global_step,
                    "Update": update
                })

            if self.cfg.algorithm == 'qd-ppo':
                log.debug(f"Finished PPO iteration {update}")

        train_elapse = time.time() - train_start
        log.debug(f'train() took {train_elapse:.2f} seconds to complete')
        if not dqd:
            log.debug("Saving checkpoint...")
            trained_models = self.vec_inference.vec_to_models()
            for i in range(num_agents):
                save_checkpoint('checkpoints', f'brax_model_{i}_checkpoint', trained_models[i],
                                self.vec_optimizer)
            # self.vec_env.stop.emit()
            log.debug("Done!")
        elif dqd:
            trained_agents = self.vec_inference.vec_to_models()
            new_params = np.array([agent.serialize() for agent in trained_agents])
            jacobian = (new_params - agent_original_params).reshape(self.cfg.num_emitters, self.cfg.num_dims + 1, -1)

            # TODO: make this work for multiple emitters
            original_agent = [Actor(self.cfg, self.obs_shape, self.action_shape).deserialize(solution_params).to(
                self.device)]
            self.vec_inference = VectorizedActor(self.cfg, original_agent, Actor,
                                                 obs_shape=self.obs_shape,
                                                 action_shape=self.action_shape)
            f, m = self.evaluate(self.vec_inference, self.multi_eval_env)
            return f.reshape(self.vec_inference.num_models, ), \
                   m.reshape(self.vec_inference.num_models, -1), \
                   jacobian

    def evaluate(self, vec_agent, vec_env):
        '''
        Evaluate all agents for one episode
        :param vec_agent: Vectorized agents for vectorized inference
        :returns: Sum rewards and measures for all agents
        '''

        total_reward = np.zeros(vec_env.num_envs)
        measures = torch.zeros(vec_env.num_envs, self.cfg.num_dims).to(self.device)
        traj_length = 0

        obs = vec_env.reset()
        obs = obs.to(self.device)
        dones = torch.BoolTensor([False for _ in range(vec_env.num_envs)])

        if self.cfg.normalize_obs:
            obs_mean = self.vec_inference.obs_normalizer.obs_rms.mean.to(self.device)
            obs_var = self.vec_inference.obs_normalizer.obs_rms.var.to(self.device)

        while not torch.all(dones):
            with torch.no_grad():
                if self.cfg.normalize_obs:
                    obs = (obs - obs_mean) / torch.sqrt(obs_var + 1e-8)
                acts, _, _ = vec_agent.get_action(obs)
                obs, rew, next_dones, infos = vec_env.step(acts.detach().cpu().numpy())
                traj_length += 1
                measures += infos['measures']  # TODO: should this be truncated?
                obs = obs.to(self.device)
                total_reward += rew.detach().cpu().numpy() * ~dones.cpu().numpy()
                dones = torch.logical_or(dones, next_dones.cpu())

        # get the final measures
        measures = measures / traj_length
        measures = measures.reshape(vec_agent.num_models, vec_env.num_envs // vec_agent.num_models, -1).mean(dim=1)

        max_reward = np.max(total_reward)
        total_reward = total_reward.reshape((vec_agent.num_models, vec_env.num_envs // vec_agent.num_models))
        total_reward = total_reward.mean(axis=1)

        # traj_lengths = infos['traj_length']

        log.debug('Finished Evaluation Step')
        log.info(f'BC on eval: {measures}')
        log.info(f'Rewards on eval: {total_reward}')
        log.info(f'Max Reward on eval: {max_reward}')
        # log.info(f'{traj_lengths=}')

        return total_reward.reshape(-1, ), measures.reshape(-1, self.cfg.num_dims).detach().cpu().numpy()
