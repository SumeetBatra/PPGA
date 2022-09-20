import torch
import numpy as np
import random
import torch.nn as nn
import wandb

from envs.VecEnv import VecEnv
from utils.utils import log

# based off of the clean-rl implementation
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_shape, action_shape: np.ndarray):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(action_shape)), std=0.01),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_shape)))

    def get_value(self, obs):
        return self.critic(obs)

    def get_action_and_value(self, obs, action=None):
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = torch.distributions.Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)


class PPO:
    def __init__(self, seed, cfg):
        self.vec_env = VecEnv(cfg.env_name,
                              num_workers=cfg.num_workers,
                              envs_per_worker=cfg.envs_per_worker)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.agent = Agent(self.vec_env.obs_shape, self.vec_env.action_space.shape).to(self.device)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=cfg.learning_rate, eps=1e-5)
        self.cfg = cfg

        # seeding
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = cfg.torch_deterministic

    def calculate_rewards(self, next_obs, next_done, rewards, values, dones, traj_len):
        # bootstrap value if not done
        with torch.no_grad():
            next_value = self.agent.get_value(next_obs).reshape(1, -1).to(self.device)
            # assume we use gae
            advantages = torch.zeros_like(rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(traj_len)):
                if t == traj_len - 1:
                    is_next_nonterminal = 1.0 - next_done
                    next_values = next_value
                else:
                    is_next_nonterminal = 1.0 - dones[t + 1]
                    next_values = values[t+1]
                is_next_nonterminal = is_next_nonterminal.to(self.device).reshape(1, -1)
                delta = rewards[t] + self.cfg.gamma * next_values * is_next_nonterminal - values[t]
                advantages[t] = lastgaelam =\
                    delta + self.cfg.gamma * self.cfg.gae_lambda * is_next_nonterminal * lastgaelam
            returns = advantages + values
        return advantages, returns

    def train(self, num_updates, traj_len=100):
        obs = torch.zeros((traj_len, self.vec_env.num_envs) + self.vec_env.obs_shape).to(self.device)
        actions = torch.zeros((traj_len, self.vec_env.num_envs) + self.vec_env.action_space.shape).to(self.device)
        logprobs = torch.zeros((traj_len, self.vec_env.num_envs)).to(self.device)
        rewards = torch.zeros((traj_len, self.vec_env.num_envs)).to(self.device)
        dones = torch.zeros((traj_len, self.vec_env.num_envs)).to(self.device)
        values = torch.zeros((traj_len, self.vec_env.num_envs)).to(self.device)

        global_step = 0
        next_obs, _, _ = self.vec_env.reset()
        next_obs = next_obs.to(self.device)
        next_done = torch.zeros(self.vec_env.num_envs).to(self.device)

        for update in range(1, num_updates + 1):
            # learning rate annealing
            if self.cfg.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.cfg.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(traj_len):
                global_step += self.vec_env.num_envs
                obs[step] = next_obs
                dones[step] = next_done.view(-1)

                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                next_obs, reward, next_done = self.vec_env.step(action.cpu().numpy())
                next_obs = next_obs.to(self.device)
                rewards[step] = reward.squeeze()

                # TODO: logging
                log.debug(f'{global_step=}')
                if self.cfg.use_wandb:
                    wandb.log({'Env Steps': global_step})

            advantages, returns = self.calculate_rewards(next_obs, next_done, rewards, values, dones, traj_len=traj_len)

            # flatten the batch
            b_obs = obs.reshape((-1,) + self.vec_env.obs_shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + self.vec_env.action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # optimize the policy and value network
            b_inds = torch.arange(self.cfg.batch_size)
            clipfracs = []
            for epoch in range(self.cfg.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.cfg.batch_size, self.cfg.minibatch_size):
                    end = start + self.cfg.minibatch_size
                    mb_inds = b_inds[start:end]  # grab enough inds for one minibatch

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds],
                                                                                       b_actions[mb_inds])
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
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.cfg.max_grad_norm)
                    self.optimizer.step()

                if self.cfg.target_kl is not None:
                    if approx_kl > self.cfg.target_kl:
                        break

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
                    "average_reward": rewards.mean(),
                })
