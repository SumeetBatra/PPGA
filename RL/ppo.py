import gym.vector
import numpy as np
import random
import time

import torch
import torch.nn as nn

import utils.utils
import wandb

from collections import deque
from functorch import vmap, combine_state_for_ensemble
from envs.vec_env import VecEnv
from envs.env import make_env
from utils.utils import log, save_checkpoint
from utils.vectorized2 import VectorizedActorCriticShared, QDVectorizedActorCriticShared
from envs.wrappers.normalize_torch import NormalizeReward, NormalizeObservation
from models.actor_critic import QDActorCriticShared, ActorCriticShared


# based off of the clean-rl implementation
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py


def make_vec_env(cfg):
    vec_env = VecEnv(cfg,
                     cfg.env_name,
                     num_workers=cfg.num_workers,
                     envs_per_worker=cfg.envs_per_worker)
    # these wrappers are applied at the vecenv level instead of the single env level
    # if cfg.normalize_obs:
    #     vec_env = NormalizeObservation(vec_env)
    # if cfg.normalize_rewards:
    #     vec_env = NormalizeReward(vec_env, gamma=cfg.gamma)
    return vec_env


def make_vec_env_for_eval(cfg, num_workers, envs_per_worker):
    vec_env = VecEnv(cfg,
                     cfg.env_name,
                     num_workers,
                     envs_per_worker)

    return vec_env


def gradient(agent):
    """Returns 1D array with gradient of all parameters in the actor."""
    return np.concatenate(
        [p.grad.cpu().detach().numpy().ravel() for p in agent.parameters()])


def gradient_linear(agent):
    return [p.grad.cpu().detach().numpy().ravel() for p in agent.actor.parameters()][0]


class PPO:
    def __init__(self, seed, cfg):
        self.vec_env = make_vec_env(cfg)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        obs_shape = self.vec_env.single_observation_space.shape
        action_shape = self.vec_env.single_action_space.shape
        agent = QDActorCriticShared(cfg, obs_shape, action_shape, cfg.num_dims).to(self.device)
        self._agent = QDVectorizedActorCriticShared(cfg, [agent], QDActorCriticShared, cfg.num_dims).to(self.device)
        # self._agent = QDActorCriticShared(cfg, obs_shape, action_shape, cfg.num_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self._agent.parameters(), lr=cfg.learning_rate, eps=1e-5)
        self.cfg = cfg

        # metrics for logging
        self.metric_last_n_window = 10
        self.episodic_returns = deque([], maxlen=self.metric_last_n_window)
        self.episodic_returns.append(0)
        self._report_interval = 5.0  # report returns every 5 seconds
        self._last_interval = 0.0

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

    @property
    def agent(self):
        return self._agent

    @agent.setter
    def agent(self, agent):
        self._agent = agent
        self.optimizer = torch.optim.Adam(self._agent.parameters(), lr=self.cfg.learning_rate, eps=1e-5)

    def calculate_rewards(self, next_obs, next_done, rewards, values, dones, rollout_length, measure_reward=False):
        # bootstrap value if not done
        with torch.no_grad():
            if measure_reward:
                next_value = self._agent.get_measure_values(next_obs).reshape(1, self.cfg.num_envs, -1).to(self.device)
            else:
                next_value = self._agent.get_value(next_obs).reshape(1, -1).to(self.device)
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

    def update(self, values, batched_data):
        b_values = values.reshape(-1)
        (b_obs, b_logprobs, b_actions, b_advantages, b_returns) = batched_data
        b_inds = torch.arange(self.cfg.batch_size)
        clipfracs = []
        for epoch in range(self.cfg.update_epochs):
            for start in range(0, self.cfg.batch_size, self.cfg.minibatch_size):
                end = start + self.cfg.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy = self._agent.get_action(b_obs[mb_inds],
                                                                b_actions[mb_inds])
                newvalue = self._agent.get_value(b_obs[mb_inds])
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

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._agent.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

            if self.cfg.target_kl is not None:
                if approx_kl > self.cfg.target_kl:
                    break

        return b_values, b_returns, pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs

    def train(self, num_updates, rollout_length):
        global_step = 0
        next_obs = self.vec_env.reset()
        next_obs = next_obs.to(self.device)
        next_done = torch.zeros(self.vec_env.num_envs).to(self.device)

        for update in range(1, num_updates + 1):
            # learning rate annealing
            if self.cfg.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.cfg.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(rollout_length):
                global_step += self.vec_env.num_envs
                if self.cfg.normalize_obs:
                    next_obs = self._agent.vec_normalize_obs(next_obs)
                self.obs[step] = next_obs
                self.dones[step] = next_done.view(-1)

                with torch.no_grad():
                    action, logprob, _ = self._agent.get_action(next_obs)
                    value = self._agent.get_value(next_obs)
                    self.values[step] = value.flatten()
                    if self.cfg.algorithm == 'qd-ppo':
                        measure_values = self._agent.get_measure_values(next_obs)
                        self.measure_values[step] = measure_values
                self.actions[step] = action
                self.logprobs[step] = logprob

                next_obs, reward, next_done, infos = self.vec_env.step(action.cpu().numpy())
                next_obs = next_obs.to(self.device)
                if self.cfg.normalize_rewards:
                    reward = self._agent.vec_normalize_rewards(reward, next_done)
                self.rewards[step] = reward.squeeze()

                measures = infos['measures']
                self.measures[step] = measures

                # log.debug(f'{global_step=}')
                for i, done in enumerate(next_done.flatten()):
                    if done:
                        total_reward = infos['total_reward'][i]
                        log.debug(f'{total_reward=}')
                        # if total_reward < 10:
                        #     log.error(f'{total_reward=}')
                        self.episodic_returns.append(infos['total_reward'][i])

            advantages, returns = self.calculate_rewards(next_obs, next_done, self.rewards, self.values, self.dones,
                                                         rollout_length=self.cfg.rollout_length)
            # flatten the batch
            b_obs = self.obs.reshape((-1,) + self.vec_env.single_observation_space.shape)
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape((-1,) + self.vec_env.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)

            (b_values, b_returns, pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs) = self.update(
                self.values, (b_obs, b_logprobs, b_actions, b_advantages, b_returns))

            #########################
            ### TESTING #############
            if self.cfg.algorithm == 'qd-ppo':
                next_done_repeated = torch.repeat_interleave(next_done.unsqueeze(1), repeats=self.cfg.num_dims, dim=-1)
                dones_repeated = torch.repeat_interleave(self.dones.unsqueeze(2), repeats=self.cfg.num_dims, dim=-1)
                m_advantages, m_returns = self.calculate_rewards(next_obs, next_done_repeated, self.measures,
                                                                 self.measure_values, dones_repeated,
                                                                 rollout_length=self.cfg.rollout_length,
                                                                 measure_reward=True)
                bm_advantages = m_advantages.reshape(-1, self.cfg.num_dims)
                bm_returns = m_returns.reshape(-1, self.cfg.num_dims)
                for i in range(self.cfg.num_dims):
                    _ = self.update(self.measure_values[i], (b_obs, b_logprobs, b_actions, bm_advantages[i], bm_returns[i]))

            #########################
            #########################

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            if self.cfg.use_wandb:
                wandb.log({
                    "charts/learning_rate": self.optimizer.param_groups[0]['lr'],
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
        log.debug("Saving checkpoint...")
        save_checkpoint('checkpoints', 'checkpoint0', self._agent, self.optimizer)
        self.vec_env.stop.emit()
        log.debug("Done!")

    def evaluate(self, vec_agent):
        '''
        Evaluate all agents for one episode using deterministic actions and collect measures
        :param vec_agent: Vectorized agents for vectorized inference
        :returns: Sum rewards and measures for all agents
        '''

        # TODO: untested and definitely doesn't work. Need to implement all interfaces

        # kill the previous processes to save memory
        self.vec_env.close()
        # resize vec env s.t. there is one env per agent
        self.vec_env = make_vec_env_for_eval(self.cfg, vec_agent.num_models, envs_per_worker=1)

        total_reward = np.zeros((vec_agent.num_models,))
        measures = np.zeros((self.cfg.num_dims, vec_agent.num_models))

        obs = self.vec_env.reset()
        obs = obs.to(self.device)
        dones = torch.BoolTensor([False for _ in range(self.vec_env.num_envs)])

        while not all(dones):
            acts = vec_agent(obs)
            obs, rew, next_dones, infos = self.vec_env.step(acts.detach().cpu().numpy())
            obs = obs.to(self.device)
            total_reward += rew.detach().cpu().numpy()
            dones = torch.logical_or(dones, next_dones)

        # get the final measures stored in the infos from the final timestep
        measures = infos['bc'].detach().cpu().numpy()

        log.debug('Finished Evaluation Step')

        return total_reward.reshape(-1, ), measures.reshape(-1, self.cfg.num_dims)

    def evaluate_lander_vectorized(self, weights, num_steps):
        obs_shape, action_shape = self.vec_env.single_observation_space.shape, self.vec_env.single_action_space.shape
        agents = [LinearPolicy(obs_shape, action_shape).to(self.device) for _ in range(len(weights))]
        for agent, w in zip(agents, weights):
            agent.update_weights(w.T)
        fmodel, params, buffers = combine_state_for_ensemble(agents)

        # deterministic policies
        obs = self.vec_env.reset()
        obs = torch.from_numpy(obs).to(self.device)
        # obs = obs.repeat_interleave(agents, dim=1)
        total_reward = np.zeros((len(agents),))

        measures = np.zeros((2, self.vec_env.num_envs))
        measures_filled = torch.zeros(self.vec_env.num_envs).to(torch.bool)
        for step in range(num_steps):
            logits = vmap(fmodel)(params, buffers, obs)
            obs, rew, dones, infos = self.vec_env.step(logits.detach().cpu().numpy())
            obs = torch.from_numpy(obs).to(self.device)
            total_reward += rew

            for i, info in enumerate(infos):
                bc = info['bc']
                if bc[0] is not None and not measures_filled[i]:
                    measures[:, i] = bc
                    measures_filled[i] = True

            # log.debug(f'Evaluation Step: {step}')

        # If the lunar lander did not land, set the x-pos to the one from the final
        # timestep, and set the y-vel to the max y-vel (we use min since the lander
        # goes down).
        for i, info in enumerate(infos):
            if not measures_filled[i]:
                measures[:, i] = torch.tensor([info['x_pos'], info['y_vel']])

        log.debug('Finished Evaluation Step')
        return total_reward.reshape((-1,)), measures.reshape((-1, 2))
