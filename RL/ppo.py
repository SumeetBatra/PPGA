from typing import List

import numpy as np
import random
import copy

import torch
import time
import torch.nn as nn
import wandb
from collections import deque
from utils.utilities import log, save_checkpoint
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
        self.qd_critic = critic
        self.vec_inference = VectorizedActor(cfg, self._agents, Actor, obs_shape=self.obs_shape,
                                             action_shape=self.action_shape).to(self.device)
        self.vec_optimizer = torch.optim.Adam(self.vec_inference.parameters(), lr=cfg.learning_rate, eps=1e-5)
        self.qd_critic_optim = torch.optim.Adam(self.qd_critic.parameters(), lr=cfg.learning_rate, eps=1e-5)
        self.cfg = cfg

        # critic for moving the mean solution point
        self.mean_critic = Critic(self.obs_shape).to(self.device)
        self.mean_critic_optim = torch.optim.Adam(self.mean_critic.parameters(), lr=cfg.learning_rate, eps=1e-5)

        # metrics for logging
        self.metric_last_n_window = 10
        self.episodic_returns = deque([], maxlen=self.metric_last_n_window)
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

        next_obs = self.vec_env.reset()
        self.next_obs = next_obs.to(self.device)
        self.next_done = torch.zeros(self.vec_env.num_envs).to(self.device)

        # for moving the mean solution point w/ ppo
        self._grad_coeffs = torch.zeros(cfg.num_dims + 1).to(self.device)
        self._grad_coeffs[0] = 1.0  # default grad coefficients optimizes objective only
        self.obs_measure_coeffs = torch.zeros((cfg.rollout_length, self.vec_env.num_envs,
                                               self.vec_env.single_observation_space.shape[
                                                   0] + self.cfg.num_dims + 1)).to(self.device)

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

    @property
    def grad_coeffs(self):
        return self._grad_coeffs

    @grad_coeffs.setter
    def grad_coeffs(self, coeffs):
        if isinstance(coeffs, np.ndarray):
            coeffs = torch.tensor(coeffs).to(self.device)
        assert isinstance(coeffs, torch.Tensor), "grad coefficients should be a pytorch tensor"
        repeats = self.cfg.num_envs // coeffs.shape[0]
        coeffs = torch.repeat_interleave(coeffs, dim=0, repeats=repeats)
        self._grad_coeffs = coeffs

    def update_critics(self, critics_list: List[Critic]):
        self.qd_critic = QDCritic(self.obs_shape, measure_dim=self.cfg.num_dims, critics_list=critics_list).to(
            self.device)
        self.qd_critic_optim = torch.optim.Adam(self.qd_critic.parameters(), lr=self.cfg.learning_rate, eps=1e-5)

    def calculate_rewards(self, next_obs, next_done, rewards, values, dones, rollout_length, calculate_dqd_gradients=False,
                          move_mean_agent=False):
        # bootstrap value if not done
        with torch.no_grad():
            if calculate_dqd_gradients:
                next_obs = next_obs.reshape(self.cfg.num_dims + 1, self.cfg.num_envs // (self.cfg.num_dims + 1), -1)
                next_value = []
                for i, obs, in enumerate(next_obs):
                    next_value.append(self.qd_critic.get_value_at(obs, dim=i))
                next_value = torch.cat(next_value).reshape(1, -1).to(self.device)
            elif move_mean_agent:
                next_value = self.mean_critic.get_value(next_obs).reshape(1, -1).to(self.device)
            else:
                # standard ppo
                next_value = self.qd_critic.get_value(next_obs).reshape(1, -1).to(self.device)
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
                if calculate_dqd_gradients:
                    is_next_nonterminal = is_next_nonterminal.to(self.device)
                else:
                    is_next_nonterminal = is_next_nonterminal.to(self.device).reshape(1, -1)
                delta = rewards[t] + self.cfg.gamma * next_values * is_next_nonterminal - values[t]
                advantages[t] = lastgaelam = \
                    delta + self.cfg.gamma * self.cfg.gae_lambda * is_next_nonterminal * lastgaelam
            returns = advantages + values
        return advantages, returns

    def batch_update(self, values, batched_data, calculate_dqd_gradients=False, move_mean_agent=False):
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

                if calculate_dqd_gradients:
                    newvalue = []
                    for i in range(self.cfg.num_dims + 1):
                        newvalue.append(self.qd_critic.get_value_at(b_obs[i, mb_inds], dim=i))
                    newvalue = torch.cat(newvalue).to(self.device)
                elif move_mean_agent:
                    newvalue = self.mean_critic.get_value(b_obs[:, mb_inds].reshape(-1, obs_dim))
                else:
                    # standard ppo
                    newvalue = self.qd_critic.get_value(b_obs[:, mb_inds].reshape(-1, obs_dim))

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
                    v_loss = v_loss_max.mean()
                else:
                    v_loss = ((newvalue - b_returns[:, mb_inds].flatten()) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.cfg.entropy_coef * entropy_loss + v_loss * self.cfg.vf_coef

                for (p1, p2, p3) in zip(self.vec_inference.parameters(), self.mean_critic.parameters(), self.qd_critic.parameters()):
                    p1.grad = None
                    p2.grad = None
                    p3.grad = None
                loss.backward()
                nn.utils.clip_grad_norm_(self.vec_inference.parameters(), self.cfg.max_grad_norm)
                self.vec_optimizer.step()
                if move_mean_agent:
                    nn.utils.clip_grad_norm_(self.mean_critic.parameters(), self.cfg.max_grad_norm)
                    self.mean_critic_optim.step()
                else:
                    # works for standard ppo or the dqd step
                    nn.utils.clip_grad_norm_(self.qd_critic.parameters(), self.cfg.max_grad_norm)
                    self.qd_critic_optim.step()

            if self.cfg.target_kl is not None:
                if approx_kl > self.cfg.target_kl:
                    break

        return pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs

    def train(self, num_updates, rollout_length, calculate_dqd_gradients=False, move_mean_agent=False,
              negative_measure_gradients=False):
        global_step = 0
        self.next_obs = self.vec_env.reset()

        if calculate_dqd_gradients:
            # TODO: make this work for multiple emitters
            solution_params = self._agents[0].serialize()
            original_obs_normalizer = None
            if self.cfg.normalize_obs:
                original_obs_normalizer = self._agents[0].obs_normalizer
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
                    # b/c of torch amp, need to convert back to float32
                    action = action.to(torch.float32)
                    if calculate_dqd_gradients:
                        next_obs = self.next_obs.reshape(num_agents, self.cfg.num_envs // num_agents, -1)
                        value = []
                        for i, obs in enumerate(next_obs):
                            value.append(self.qd_critic.get_value_at(obs, i))
                        value = torch.cat(value).reshape(-1).to(self.device)
                    elif move_mean_agent:
                        value = self.mean_critic.get_value(self.next_obs)
                    else:
                        # standard ppo. Maintains backwards compatibility
                        value = self.qd_critic.get_value(self.next_obs)
                self.values[step] = value.flatten()
                self.actions[step] = action
                self.logprobs[step] = logprob

                self.next_obs, reward, self.next_done, infos = self.vec_env.step(action)
                measures = -infos['measures'] if negative_measure_gradients else infos['measures']
                self.measures[step] = measures
                if move_mean_agent:
                    with torch.no_grad():
                        rew_measures = torch.cat((reward.unsqueeze(1), measures), dim=1)
                        rew_measures *= self._grad_coeffs
                        reward = rew_measures.sum(dim=1)
                reward = reward.cpu()
                self.total_rewards += reward
                self.next_obs = self.next_obs.to(self.device)
                if self.cfg.normalize_rewards:
                    reward = self.vec_inference.vec_normalize_rewards(reward, self.next_done)
                self.rewards[step] = reward.squeeze()

                if not calculate_dqd_gradients and not move_mean_agent:
                    # TODO: move this to a separate process
                    if self.num_intervals % self._report_interval == 0:
                        for i, done in enumerate(self.next_done.flatten()):
                            if done:
                                total_reward = self.total_rewards[i].clone()
                                # log.debug(f'{total_reward=}')
                                self.episodic_returns.append(total_reward)
                                self.total_rewards[i] = 0
                    self.num_intervals += 1

            if calculate_dqd_gradients:
                with torch.no_grad():
                    envs_per_dim = self.cfg.num_envs // (self.cfg.num_dims + 1)
                    mask = torch.eye(self.cfg.num_dims + 1)
                    mask = torch.repeat_interleave(mask, dim=0, repeats=envs_per_dim).unsqueeze(dim=0).to(self.device)

                    # concat the reward w/ measures and mask appropriately
                    rew_measures = torch.cat((self.rewards.unsqueeze(dim=2), self.measures), dim=2)
                    rew_measures = (rew_measures * mask).sum(dim=2)
                advantages, returns = self.calculate_rewards(self.next_obs, self.next_done, rew_measures,
                                                             self.values, self.dones,
                                                             rollout_length=rollout_length, calculate_dqd_gradients=True)
            else:
                advantages, returns = self.calculate_rewards(self.next_obs, self.next_done, self.rewards, self.values,
                                                             self.dones, rollout_length=rollout_length,
                                                             move_mean_agent=move_mean_agent)
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
                                                                                                      calculate_dqd_gradients=calculate_dqd_gradients,
                                                                                                      move_mean_agent=move_mean_agent)

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            avg_log_stddev = self.vec_inference.actor_logstd.mean().detach().cpu().numpy()
            avg_obj_magnitude = self.rewards.mean()

            if self.cfg.use_wandb:
                wandb.log({
                    "charts/actor_avg_logstd": avg_log_stddev,
                    "charts/average_rew_magnitude": avg_obj_magnitude,
                    f"losses/{move_mean_agent=}/value_loss": v_loss.item(),
                    "losses/policy_loss": pg_loss.item(),
                    "losses/entropy": entropy_loss.item(),
                    "losses/old_approx_kl": old_approx_kl.item(),
                    "losses/approx_kl": approx_kl.item(),
                    "losses/clipfrac": np.mean(clipfracs),
                    "losses/explained_variance": explained_var,
                    "Env step": global_step,
                    "Update": update
                })

                if len(self.episodic_returns):
                    # only log if we have something to report
                    wandb.log({
                        f"Average Episodic Reward": sum(self.episodic_returns) / len(self.episodic_returns),
                        "Env step": global_step,
                        "Update": update
                    })

        train_elapse = time.time() - train_start
        log.debug(f'train() took {train_elapse:.2f} seconds to complete')
        fps = global_step / train_elapse
        log.debug(f'FPS: {fps:.2f}')
        if self.cfg.use_wandb:
            wandb.log({'FPS: ': fps})
        if not calculate_dqd_gradients and not move_mean_agent:
            # standard ppo
            log.debug("Saving checkpoint...")
            trained_models = self.vec_inference.vec_to_models()
            for i in range(num_agents):
                save_checkpoint('checkpoints', f'{self.cfg.env_name}_{self.cfg.env_type}_model_{i}_checkpoint',
                                trained_models[i],
                                self.vec_optimizer)
            log.debug("Done!")
        elif calculate_dqd_gradients:
            trained_agents = self.vec_inference.vec_to_models()
            new_params = np.array([agent.serialize() for agent in trained_agents])
            jacobian = (new_params - agent_original_params).reshape(self.cfg.num_emitters, self.cfg.num_dims + 1, -1)

            # TODO: make this work for multiple emitters
            original_agent = [Actor(self.cfg, self.obs_shape, self.action_shape).deserialize(solution_params).to(
                self.device)]
            self.vec_inference = VectorizedActor(self.cfg, original_agent, Actor,
                                                 obs_shape=self.obs_shape,
                                                 action_shape=self.action_shape)
            f, m, metadata = self.evaluate(self.vec_inference, self.vec_env, obs_normalizer=original_obs_normalizer)
            return f.reshape(self.vec_inference.num_models, ), \
                m.reshape(self.vec_inference.num_models, -1), \
                jacobian, \
                metadata

    def evaluate(self, vec_agent, vec_env, verbose=True, obs_normalizer=None):
        '''
        Evaluate all agents for one episode
        :param vec_agent: Vectorized agents for vectorized inference
        :returns: Sum rewards and measures for all agents
        '''

        total_reward = np.zeros(vec_env.num_envs)
        traj_length = 0
        num_steps = 1000

        obs = vec_env.reset()
        obs = obs.to(self.device)
        dones = torch.BoolTensor([False for _ in range(vec_env.num_envs)])
        all_dones = torch.zeros((num_steps, vec_env.num_envs)).to(self.device)
        measures_acc = torch.zeros((num_steps, vec_env.num_envs, self.cfg.num_dims)).to(self.device)
        measures = torch.zeros((vec_env.num_envs, self.cfg.num_dims)).to(self.device)

        if self.cfg.normalize_obs and obs_normalizer is not None:
            obs_mean = obs_normalizer.obs_rms.mean.to(self.device)
            obs_var = obs_normalizer.obs_rms.var.to(self.device)

        while not torch.all(dones):
            with torch.no_grad():
                if self.cfg.normalize_obs:
                    obs = (obs - obs_mean) / torch.sqrt(obs_var + 1e-8)
                acts, _, _ = vec_agent.get_action(obs)
                acts = acts.to(torch.float32)
                obs, rew, next_dones, infos = vec_env.step(acts)
                measures_acc[traj_length] = infos['measures']
                obs = obs.to(self.device)
                total_reward += rew.detach().cpu().numpy() * ~dones.cpu().numpy()
                dones = torch.logical_or(dones, next_dones.cpu())
                all_dones[traj_length] = dones.long().clone()
                traj_length += 1

        # the first done in each env is where that trajectory ends
        traj_lengths = torch.argmax(all_dones, dim=0) + 1
        # TODO: figure out how to vectorize this
        for i in range(vec_env.num_envs):
            measures[i] = measures_acc[:traj_lengths[i], i].sum(dim=0) / traj_lengths[i]
        measures = measures.reshape(vec_agent.num_models, vec_env.num_envs // vec_agent.num_models, -1).mean(dim=1).detach().cpu().numpy()

        total_reward = total_reward.reshape((vec_agent.num_models, vec_env.num_envs // vec_agent.num_models)).mean(
            axis=1)
        avg_traj_lengths = traj_lengths.to(torch.float32).reshape((vec_agent.num_models, vec_env.num_envs // vec_agent.num_models)).\
            mean(dim=1).cpu().numpy()
        metadata = np.array([{'traj_length': t,
                              'mean_critic_params': self.mean_critic.serialize(),
                              'qd_critic_params': self.qd_critic.serialize()}
                             for t in avg_traj_lengths]).reshape(-1,)
        max_reward = np.max(total_reward)
        min_reward = np.min(total_reward)
        mean_reward = np.mean(total_reward)
        mean_traj_length = torch.mean(traj_lengths.to(torch.float64)).detach().cpu().numpy().item()
        objective_measures = np.concatenate((total_reward.reshape(-1, 1), measures), axis=1)

        if self.cfg.normalize_obs:
            for i, data in enumerate(metadata):
                data['obs_normalizer'] = obs_normalizer

        if verbose:
            np.set_printoptions(suppress=True)
            log.debug('Finished Evaluation Step')
            log.info(f'Reward + Measures: {objective_measures}')
            log.info(f'Max Reward on eval: {max_reward}')
            log.info(f'Min Reward on eval: {min_reward}')
            log.info(f'Mean Reward across all agents: {mean_reward}')
            log.info(f'Average Trajectory Length: {mean_traj_length}')

        return total_reward.reshape(-1, ), measures.reshape(-1, self.cfg.num_dims), metadata
