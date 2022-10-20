import numpy as np
import random
import copy

import torch
import torch.nn as nn
import wandb
from collections import deque
from envs.cpu.vec_env import make_vec_env, make_vec_env_for_eval
from utils.utils import log, save_checkpoint
from models.vectorized import VectorizedActor
from models.actor_critic import Actor, Critic


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
        critic = Critic(self.obs_shape).to(self.device)
        self._critic = critic
        self.vec_inference = VectorizedActor(cfg, self._agents, Actor, obs_shape=self.obs_shape,
                                             action_shape=self.action_shape).to(self.device)
        self.actor_optimizers = [torch.optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5) for agent in
                                 self._agents]
        self.critic_optimizer = torch.optim.Adam(self._critic.parameters(), lr=cfg.learning_rate, eps=1e-5)
        self.cfg = cfg

        # single policy eval env
        single_eval_cfg = copy.deepcopy(cfg)
        single_eval_cfg.num_workers = 4
        single_eval_cfg.envs_per_worker = 1
        single_eval_cfg.envs_per_model = 4
        self.single_eval_env = make_vec_env_for_eval(single_eval_cfg, num_workers=single_eval_cfg.num_workers,
                                                     envs_per_worker=single_eval_cfg.envs_per_worker)

        # multi-policy eval
        multi_eval_cfg = copy.deepcopy(cfg)
        multi_eval_cfg.num_workers = cfg.mega_lambda
        multi_eval_cfg.envs_per_worker = 4
        multi_eval_cfg.envs_per_model = 4
        self.multi_eval_env = make_vec_env_for_eval(multi_eval_cfg, num_workers=multi_eval_cfg.num_workers,
                                                    envs_per_worker=multi_eval_cfg.envs_per_worker)

        # metrics for logging
        self.metric_last_n_window = 10
        self.episodic_returns = deque([], maxlen=self.metric_last_n_window)
        self.episodic_returns.append(0)
        self._report_interval = 5.0  # report returns every 5 seconds
        self._last_interval = 0.0
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
        return self._agents

    @agents.setter
    def agents(self, agents):
        self._agents = agents
        self.vec_inference = VectorizedActor(self.cfg, self._agents, Actor,
                                             obs_shape=self.vec_env.obs_shape,
                                             action_shape=self.vec_env.action_space.shape)

    def calculate_rewards(self, next_obs, next_done, rewards, values, dones, rollout_length, measure_reward=False):
        # bootstrap value if not done
        with torch.no_grad():
            if measure_reward:
                next_value = self.vec_inference.get_measure_values(next_obs).reshape(1, self.cfg.num_envs, -1).to(
                    self.device)
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
                if measure_reward:
                    is_next_nonterminal = is_next_nonterminal.to(self.device)
                else:
                    is_next_nonterminal = is_next_nonterminal.to(self.device).reshape(1, -1)
                delta = rewards[t] + self.cfg.gamma * next_values * is_next_nonterminal - values[t]
                advantages[t] = lastgaelam = \
                    delta + self.cfg.gamma * self.cfg.gae_lambda * is_next_nonterminal * lastgaelam
            returns = advantages + values
        return advantages, returns

    def update(self, values, batched_data, actor_optimizer, i):
        b_values = values.reshape(-1)
        (b_obs, b_logprobs, b_actions, b_advantages, b_returns) = batched_data
        b_inds = torch.arange(self.cfg.batch_size)
        clipfracs = []
        for epoch in range(self.cfg.update_epochs):
            for start in range(0, self.cfg.batch_size, self.cfg.minibatch_size):
                end = start + self.cfg.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy = self._agents[i].get_action(b_obs[mb_inds],
                                                                    b_actions[mb_inds])
                newvalue = self._critic.get_value(b_obs[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.cfg.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.cfg.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.cfg.clip_coef, 1 + self.cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # value loss
                newvalue = newvalue.view(-1)
                if self.cfg.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.cfg.clip_coef,
                        self.cfg.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.cfg.entropy_coef * entropy_loss + v_loss * self.cfg.vf_coef

                actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._agents[i].parameters(), self.cfg.max_grad_norm)
                actor_optimizer.step()
                nn.utils.clip_grad_norm_(self._critic.parameters(), self.cfg.max_grad_norm)
                self.critic_optimizer.step()

            if self.cfg.target_kl is not None:
                if approx_kl > self.cfg.target_kl:
                    break

        return pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs

    def train(self, num_updates, rollout_length):
        global_step = 0

        num_agents = len(self._agents)

        original_params = copy.deepcopy(self.vec_inference.serialize())

        for update in range(1, num_updates + 1):
            # learning rate annealing
            # TODO: not sure if we should keep this
            # if self.cfg.anneal_lr:
            #     frac = 1.0 - (update - 1.0) / num_updates
            #     lrnow = frac * self.cfg.learning_rate
            #     optimizer.param_groups[0]["lr"] = lrnow

            for step in range(rollout_length):
                global_step += self.vec_env.num_envs
                if self.cfg.normalize_obs:
                    self.next_obs = self.vec_inference.vec_normalize_obs(self.next_obs)
                self.obs[step] = self.next_obs
                self.dones[step] = self.next_done.view(-1)

                with torch.no_grad():
                    action, logprob, _ = self.vec_inference.get_action(self.next_obs)
                    value = self._critic.get_value(self.next_obs)
                    self.values[step] = value.flatten()
                    # if self.cfg.algorithm == 'qd-ppo':
                    #     measure_values = self.vec_inference.get_measure_values(self.next_obs)
                    #     self.measure_values[step] = measure_values
                self.actions[step] = action
                self.logprobs[step] = logprob

                self.next_obs, reward, self.next_done, infos = self.vec_env.step(action.cpu().numpy())
                reward = reward.cpu()
                self.total_rewards += reward
                self.next_obs = self.next_obs.to(self.device)
                if self.cfg.normalize_rewards:
                    reward = self.vec_inference.vec_normalize_rewards(reward, self.next_done)
                self.rewards[step] = reward.squeeze()

                # measures = infos['measures']
                # self.measures[step] = measures

                ########################################
                ## EXPERIMENTAL
                # m_reward = (measures.unsqueeze(dim=1).reshape(len(self._agents), -1, self.cfg.num_dims) *
                #             self.vec_inference.measure_coeffs.unsqueeze(dim=1)).sum(dim=2).reshape(-1)
                # reward += m_reward
                ########################################

                for i, done in enumerate(self.next_done.flatten()):
                    if done:
                        total_reward = self.total_rewards[i].clone()
                        log.debug(f'{total_reward=}')
                        self.episodic_returns.append(total_reward)
                        self.total_rewards[i] = 0

            advantages, returns = self.calculate_rewards(self.next_obs, self.next_done, self.rewards, self.values,
                                                         self.dones,
                                                         rollout_length=self.cfg.rollout_length)
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
                    self.actor_optimizers[i], i)
                if self.cfg.algorithm == 'qd-ppo':
                    obj_grad = self.vec_inference.serialize() - original_params

            # update vec inference
            # TODO: create an update_params() method instead of recreating vec_inf each time
            self.vec_inference = VectorizedActor(self.cfg, self._agents, Actor,
                                                 obs_shape=self.obs_shape,
                                                 action_shape=self.action_shape)

            #########################
            ### TESTING #############
            # if self.cfg.algorithm == 'qd-ppo':
            #
            #     m_grads = []
            #     next_done_repeated = torch.repeat_interleave(next_done.unsqueeze(1), repeats=self.cfg.num_dims, dim=-1)
            #     dones_repeated = torch.repeat_interleave(self.dones.unsqueeze(2), repeats=self.cfg.num_dims, dim=-1)
            #     m_advantages, m_returns = self.calculate_rewards(next_obs, next_done_repeated, self.measures,
            #                                                      self.measure_values, dones_repeated,
            #                                                      rollout_length=self.cfg.rollout_length,
            #                                                      measure_reward=True)
            #     bm_advantages = m_advantages.reshape(-1, self.cfg.num_dims)
            #     bm_returns = m_returns.reshape(-1, self.cfg.num_dims)
            #     for i in range(self.cfg.num_dims):
            #         # reset agent back to original solution point
            #         self._agent.deserialize(original_params).to(self.device)
            #         _ = self.update(self.measure_values[:, :, i],
            #                         (b_obs, b_logprobs, b_actions, bm_advantages[:, i], bm_returns[:, i]))
            #         m_grad = self._agent.serialize() - original_params
            #         m_grads.append(m_grad)

            #########################
            #########################
            # fake the gradients for testing
            m_grads = np.zeros((self.vec_inference.num_models, self.cfg.num_dims, len(original_params)))

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            if self.cfg.use_wandb:
                wandb.log({
                    "charts/learning_rate": self.actor_optimizers[0].param_groups[0]['lr'],
                    "losses/value_loss": v_loss.item(),
                    "losses/policy_loss": pg_loss.item(),
                    "losses/entropy": entropy_loss.item(),
                    "losses/old_approx_kl": old_approx_kl.item(),
                    "losses/approx_kl": approx_kl.item(),
                    "losses/clipfrac": np.mean(clipfracs),
                    "losses/explained_variance": explained_var,
                    f"Average Episodic Reward": sum(self.episodic_returns) / len(self.episodic_returns),
                    "Env step": global_step
                })

            if self.cfg.algorithm == 'qd-ppo':
                log.debug(f"Finished PPO iteration {update}")

        if self.cfg.algorithm == 'ppo':
            log.debug("Saving checkpoint...")
            trained_models = self.vec_inference.vec_to_models()
            for i in range(num_agents):
                save_checkpoint('checkpoints', f'brax_model_{i}_checkpoint', self._agents[i], self.actor_optimizers[i])
            # self.vec_env.stop.emit()
            log.debug("Done!")
        else:
            m_grads = np.concatenate(m_grads, axis=0).reshape(self.cfg.num_dims, -1)
            f, m = self.evaluate(self.vec_inference, self.single_eval_env)
            return f.reshape(self.vec_inference.num_models, ), \
                   obj_grad.reshape(self.vec_inference.num_models, -1), \
                   m.reshape(self.vec_inference.num_models, -1), \
                   m_grads.reshape(self.vec_inference.num_models, self.cfg.num_dims, -1)

    def evaluate(self, vec_agent, vec_env):
        '''
        Evaluate all agents for one episode
        :param vec_agent: Vectorized agents for vectorized inference
        :returns: Sum rewards and measures for all agents
        '''

        total_reward = np.zeros(vec_env.num_envs)

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
                obs = obs.to(self.device)
                total_reward += rew.detach().cpu().numpy() * ~dones.cpu().numpy()
                dones = torch.logical_or(dones, next_dones.cpu())

        # get the final measures stored in the infos from the final timestep
        # measures = infos['bc'].detach().cpu().numpy()
        measures = np.zeros((vec_env.num_envs, 4)).reshape(vec_agent.num_models,
                                                           vec_env.num_envs // vec_agent.num_models, -1).mean(axis=1)

        max_reward = np.max(total_reward)
        total_reward = total_reward.reshape((vec_agent.num_models, vec_env.num_envs // vec_agent.num_models))
        total_reward = total_reward.mean(axis=1)

        # traj_lengths = infos['traj_length']

        log.debug('Finished Evaluation Step')
        log.info(f'BC on eval: {measures}')
        log.info(f'Rewards on eval: {total_reward}')
        log.info(f'Max Reward on eval: {max_reward}')
        # log.info(f'{traj_lengths=}')

        return total_reward.reshape(-1, ), measures.reshape(-1, self.cfg.num_dims)
