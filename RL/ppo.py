import gym.vector
import torch
import numpy as np
import random
import torch.nn as nn

import utils.utils
import wandb

from functorch import vmap, combine_state_for_ensemble
from envs.vec_env import VecEnv
from envs.env import make_env
from utils.utils import log, save_checkpoint
from utils.vectorized2 import VectorizedPolicy
from envs.wrappers.normalize_torch import NormalizeReward, NormalizeObservation


# based off of the clean-rl implementation
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def make_vec_env(cfg):
    # vec_env = VecEnv(cfg,
    #                  cfg.env_name,
    #                  num_workers=cfg.num_workers,
    #                  envs_per_worker=cfg.envs_per_worker)
    # these wrappers are applied at the vecenv level instead of the single env level
    # vec_env = NormalizeObservation(vec_env)
    # vec_env = NormalizeReward(vec_env, gamma=cfg.gamma)

    vec_env = gym.vector.AsyncVectorEnv(
        [make_env(cfg.env_name, cfg.seed + i, cfg.gamma) for i in range(cfg.num_workers)],
        shared_memory=True
    )
    return vec_env


def gradient(agent):
    """Returns 1D array with gradient of all parameters in the actor."""
    return np.concatenate(
        [p.grad.cpu().detach().numpy().ravel() for p in agent.parameters()])


def gradient_linear(agent):
    return [p.grad.cpu().detach().numpy().ravel() for p in agent.actor.parameters()][0]


class Agent(nn.Module):
    def __init__(self, obs_shape, action_shape: np.ndarray):
        super().__init__()

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(action_shape)), std=0.01),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_shape)))

    @property
    def layers(self):
        return self.actor_mean

    def forward(self, x):
        return self.actor_mean(x)

    def get_action(self, obs, action=None):
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        cov_mat = torch.diag_embed(action_std)
        probs = torch.distributions.MultivariateNormal(action_mean, cov_mat)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()


class LinearPolicy(nn.Module):
    def __init__(self, obs_shape, action_shape):
        super().__init__()
        self.actor = layer_init(nn.Linear(np.prod(obs_shape),
                                          np.prod(action_shape)))
        self.layer = nn.Sequential(*[self.actor])
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_shape)))

    def get_action(self, obs, action=None):
        action_mean = self.actor(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        cov_mat = torch.diag_embed(action_std)
        probs = torch.distributions.MultivariateNormal(action_mean, cov_mat)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()

    @property
    def layers(self):
        return self.layer

    def forward(self, x):
        return self.actor(x)

    def update_weights(self, weights):
        self.actor.weight.data = weights


class GlobalCritic(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()
        self.m_dim = 2  # dimensionality of the measures
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod() + self.m_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_value(self, obs):
        return self.critic(obs)


class PPO:
    def __init__(self, seed, cfg, agent=None):
        self.vec_env = make_vec_env(cfg)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if agent:
            self._agent = agent
        else:
            # self._agent = Agent(self.vec_env.single_observation_space.shape, self.vec_env.single_action_space.shape).to(
            #     self.device)
            obs_shape, action_shape = self.vec_env.single_observation_space.shape, \
                                      self.vec_env.single_action_space.shape
            agents = [
                LinearPolicy(obs_shape, action_shape).to(self.device)
                for _ in range(cfg.num_emitters)
            ]
            self._agent = VectorizedPolicy(agents, LinearPolicy, obs_shape=obs_shape, action_shape=action_shape) \
                .to(self.device)
            self._critic = GlobalCritic(self.vec_env.single_observation_space.shape).to(self.device)

        self.actor_optim = torch.optim.Adam(self._agent.parameters(), lr=cfg.learning_rate, eps=1e-5)
        self.critic_optim = torch.optim.Adam(self._critic.parameters(), lr=cfg.learning_rate, eps=1e-5)
        self.cfg = cfg

        # seeding
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = cfg.torch_deterministic

    @property
    def agent(self):
        '''
        Parameterized policy that will interact with the env
        '''
        return self._agent

    @agent.setter
    def agent(self, agent):
        '''
        Update the policy
        :param agent:  New policy
        '''
        self._agent = agent

    def calculate_rewards(self, next_obs, next_done, rewards, values, dones, traj_len):
        # bootstrap value if not done
        with torch.no_grad():
            next_value = self._critic.get_value(next_obs).reshape(1, -1).to(self.device)
            # assume we use gae
            advantages = torch.zeros_like(rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(traj_len)):
                if t == traj_len - 1:
                    is_next_nonterminal = 1.0 - next_done.long()
                    next_values = next_value
                else:
                    is_next_nonterminal = 1.0 - dones[t + 1]
                    next_values = values[t + 1]
                is_next_nonterminal = is_next_nonterminal.to(self.device).reshape(1, -1)
                delta = rewards[t] + self.cfg.gamma * next_values * is_next_nonterminal - values[t]
                advantages[t] = lastgaelam = \
                    delta + self.cfg.gamma * self.cfg.gae_lambda * is_next_nonterminal * lastgaelam
            returns = advantages + values
        return advantages, returns

    def train(self, num_updates, traj_len):
        num_updates = 1
        self.actor_optim = torch.optim.Adam(self._agent.parameters(), lr=self.cfg.learning_rate, eps=1e-5)

        obs_shape = self.vec_env.single_observation_space.shape
        obs_measure_shape = (obs_shape[0] + 2,)
        obs = torch.zeros((traj_len, self.vec_env.num_envs) +
                          self.vec_env.single_observation_space.shape).to(self.device)
        obs_measures = torch.zeros((traj_len, self.vec_env.num_envs) +
                                   obs_measure_shape).to(self.device)
        actions = torch.zeros((traj_len, self.vec_env.num_envs) +
                              self.vec_env.single_action_space.shape).to(self.device)
        logprobs = torch.zeros((traj_len, self.vec_env.num_envs)).to(self.device)
        rewards = torch.zeros((traj_len, self.vec_env.num_envs)).to(self.device)
        dones = torch.zeros((traj_len, self.vec_env.num_envs)).to(self.device)
        values = torch.zeros((traj_len, self.vec_env.num_envs)).to(self.device)
        measures = torch.zeros((2, self.vec_env.num_envs), requires_grad=True).to(self.device)
        measures_filled = torch.zeros(self.vec_env.num_envs).to(torch.bool)
        total_reward = 0.0

        # original params so we can measure f's jacobian later
        original_theta = np.array([p.detach().cpu().numpy().ravel() for p in self.agent.parameters()][1])
        # 2 measures for lunar lander
        m_jacobians = np.array([[np.zeros_like(p.detach().cpu().numpy().ravel())
                                 for p in self._agent.actor.parameters()][0] for _ in range(2)])

        global_step = 0
        next_obs = self.vec_env.reset()
        next_obs = torch.from_numpy(next_obs).to(self.device)
        # next_obs = next_obs.to(self.device)
        next_done = torch.zeros(self.vec_env.num_envs).to(self.device)

        for update in range(1, num_updates + 1):
            # learning rate annealing
            if self.cfg.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.cfg.learning_rate
                self.actor_optim.param_groups[0]["lr"] = lrnow
                self.critic_optim.param_groups[0]["lr"] = lrnow

            for step in range(traj_len):
                global_step += self.vec_env.num_envs
                obs[step] = next_obs
                dones[step] = next_done.view(-1)

                with torch.no_grad():
                    action, logprob, _ = self._agent.get_action(next_obs, self._agent.action_logstds)
                    # value = self._critic.get_value(next_obs)
                    # values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                next_obs, reward, next_done, infos = self.vec_env.step(action.cpu().numpy())
                next_done = torch.from_numpy(next_done).to(self.device)
                next_obs = torch.from_numpy(next_obs).to(self.device)
                rewards[step] = torch.from_numpy(reward).squeeze()
                # rewards[step] = reward.squeeze()

                for i, info in enumerate(infos):
                    bc = info['desc']
                    if bc[0] is not None and not measures_filled[i]:
                        measures[:, i] = torch.tensor(bc).to(self.device)
                        measures_filled[i] = True

            # If the lunar lander did not land, set the x-pos to the one from the final
            # timestep, and set the y-vel to the max y-vel (we use min since the lander
            # goes down).
            for i, info in enumerate(infos):
                if not measures_filled[i]:
                    measures[:, i] = torch.tensor([info['x_pos'], info['y_vel']]).to(self.device)

            for step in range(traj_len):
                with torch.no_grad():
                    next_obs = obs[step]
                    all_obs = torch.cat((next_obs, measures.T), dim=1)
                    obs_measures[step] = all_obs
                    value = self._critic.get_value(all_obs)
                    values[step] = value.flatten()

                # TODO: logging
                # log.debug(f'{global_step=}')
            advantages, returns = self.calculate_rewards(all_obs, next_done, rewards, values, dones, traj_len=traj_len)

            # flatten the batch
            b_obs = obs.reshape((-1,) + self.vec_env.single_observation_space.shape)
            b_obs_measures = obs_measures.reshape((-1,) + (self.vec_env.single_observation_space.shape[0] + 2,))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + self.vec_env.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # measures for QD
            measures_avg = torch.mean(measures, dim=1).reshape(-1, 1)

            # optimize the policy and value network
            b_inds = torch.arange(self.cfg.batch_size)
            clipfracs = []
            for epoch in range(self.cfg.update_epochs):
                # np.random.shuffle(b_inds)
                for start in range(0, self.cfg.batch_size, self.cfg.minibatch_size):
                    end = start + self.cfg.minibatch_size
                    mb_inds = b_inds[start:end]  # grab enough inds for one minibatch
                    # TODO: fix this hack
                    mb_size = len(mb_inds)
                    if mb_size % self._agent.num_models != 0:
                        r = mb_size // self._agent.num_models
                        div = int(self._agent.num_models * r)
                        mb_inds = mb_inds[:div]

                    _, newlogprob, entropy = self._agent.get_action(b_obs[mb_inds],
                                                                    self._agent.action_logstds,
                                                                    actions=b_actions[mb_inds])
                    newvalue = self._critic.get_value(b_obs_measures[mb_inds])
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

                    # measure loss
                    measures_loss = -measures_avg * ratio
                    measures_loss = torch.mean(measures_loss, dim=1)

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

                    self.actor_optim.zero_grad()
                    self.critic_optim.zero_grad()
                    for i, m_loss in enumerate(measures_loss):
                        torch.autograd.backward(m_loss, retain_graph=True, inputs=[p for p in self.agent.parameters()])
                        m_jacobians[i] += gradient_linear(self.agent)
                        self.actor_optim.zero_grad()

                    loss.backward()
                    nn.utils.clip_grad_norm_(self._agent.parameters(), self.cfg.max_grad_norm)
                    nn.utils.clip_grad_norm_(self._critic.parameters(), self.cfg.max_grad_norm)
                    self.actor_optim.step()
                    self.critic_optim.step()

                if self.cfg.target_kl is not None:
                    if approx_kl > self.cfg.target_kl:
                        break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # avg_reward = torch.sum(rewards) / self.vec_env.num_envs
            avg_reward = torch.mean(rewards.reshape(traj_len, -1, 6), dim=(0, 2))
            total_reward += avg_reward.detach().cpu().numpy()

            if self.cfg.use_wandb:
                wandb.log({
                    "charts/learning_rate": self.actor_optim.param_groups[0]['lr'],
                    "losses/value_loss": v_loss.item(),
                    "losses/policy_loss": pg_loss.item(),
                    "losses/entropy": entropy_loss.item(),
                    "losses/old_approx_kl": old_approx_kl.item(),
                    "losses/approx_kl": approx_kl.item(),
                    "losses/clipfrac": np.mean(clipfracs),
                    "losses/explained_variance": explained_var,
                    "RL/average_reward": torch.sum(rewards) / self.vec_env.num_envs,
                    "Env step": global_step
                })
        # log.debug("Saving checkpoint...")
        # save_checkpoint('checkpoints', 'checkpoint0', self._agent, self.actor_optim)
        # self.vec_env.stop.emit()
        # self.vec_env.close()
        log.debug("Finished PPO training step!")
        f_jacobian = np.array([p.detach().cpu().numpy().ravel() for p in self.agent.parameters()][1]) - original_theta
        m_jacobians = m_jacobians.reshape((self.cfg.num_emitters, 2, -1))
        measures = measures.reshape(self.cfg.num_emitters, 2, -1).mean(dim=2).reshape(5, -1).detach().cpu().numpy()
        return total_reward.reshape((-1,)), \
               f_jacobian.reshape(5, -1), \
               measures, \
               m_jacobians

    def evaluate_lander(self, agents, num_steps):
        '''
        Evaluate the objective and measures on a batch of agents
        :param num_steps: num steps to evaluate against
        :param agents: batch of policies
        :return: objective and measure scores for all agents
        '''
        # TODO: eventually this will be nn-policies. Will need to vmap them
        #  or use the vectorized-inference implementation

        # deterministic policies
        obs = self.vec_env.reset()
        obs = torch.from_numpy(obs).to(self.device)
        # obs = obs.repeat_interleave(agents, dim=1)
        total_reward = np.zeros((len(agents),))

        measures = np.zeros((2, self.vec_env.num_envs))
        measures_filled = torch.zeros(self.vec_env.num_envs).to(torch.bool)
        for step in range(num_steps):
            all_logits = []
            for i, agent in enumerate(agents):
                logits = obs[i] @ agent
                all_logits.append(logits)
            all_logits = torch.concat(all_logits).reshape(-1, 2)
            obs, rew, dones, infos = self.vec_env.step(all_logits.detach().cpu().numpy())
            obs = torch.from_numpy(obs).to(self.device)
            total_reward += rew

            for i, info in enumerate(infos):
                bc = info['desc']
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
                bc = info['desc']
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
