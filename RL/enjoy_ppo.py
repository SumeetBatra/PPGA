from attrdict import AttrDict
from RL.ppo import *
from utils.utils import log
from envs.cpu.env import make_env
from envs.brax_custom.gpu_env import make_vec_env_brax
from models.actor_critic import Actor

from IPython.display import HTML, Image
from IPython.display import display
from brax.io import html, image
from brax import envs
from jax import numpy as jnp


def enjoy():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # env = gym.make("QDAntBulletEnv-v0", render=True)
    # env.seed(8679)
    env = make_env('QDAntBulletEnv-v0', seed=0, gamma=0.99)()
    env.render()
    obs_shape, action_shape = env.observation_space.shape, env.action_space.shape
    cp_path = "checkpoints/brax_model_0_checkpoint"
    model_state_dict = torch.load(cp_path)['model_state_dict']
    cfg = {'num_workers': 1, 'envs_per_worker': 1, 'normalize_obs': False, 'normalize_rewards': True, 'num_dims': 4, 'envs_per_model': 1}
    cfg = AttrDict(cfg)
    agent = Actor(cfg, obs_shape, action_shape).to(device)
    # agent = QDVectorizedActorCriticShared(cfg, [agent], QDActorCriticShared, measure_dims=4)
    model_state_dict['_actor_logstd'] = model_state_dict['_actor_logstd'].reshape(1, -1)
    agent.load_state_dict(model_state_dict)

    if cfg.normalize_obs:
        obs_mean, obs_var = agent.obs_normalizer.obs_rms.mean, agent.obs_normalizer.obs_rms.var
        obs_mean = obs_mean.to(device)
        obs_var = obs_var.to(device)

    obs = env.reset()
    obs = torch.from_numpy(obs).to(device).reshape(1, -1)
    done = False
    total_reward = 0
    while not done:
        if cfg.normalize_obs:
            obs = (obs - obs_mean) / torch.sqrt(obs_var + 1e-8)
        act, _, _ = agent.get_action(obs)
        # act = agent(obs)
        act = act.squeeze()
        obs, rew, done, info = env.step(act.detach().cpu().numpy())
        log.debug(f'{rew=}')
        total_reward += rew
        obs = torch.from_numpy(obs).to(device).reshape(1, -1)
        # env.render()
    fitness = total_reward
    log.debug(f'{fitness=}')
    measures = info['bc']
    log.debug(f'{measures=}')
    env.close()


def enjoy_brax():
    cfg = {'env_name': 'ant', 'env_batch_size': None, 'normalize_obs': False, 'normalize_rewards': True,
           'num_dims': 4, 'envs_per_model': 1, 'seed': 0}
    cfg = AttrDict(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = make_vec_env_brax(cfg)

    obs_shape, action_shape = env.observation_space.shape, env.action_space.shape
    agent = QDActorCriticShared(cfg, obs_shape, action_shape, 4).to(device)
    cp_path = "checkpoints/model_0_checkpoint"
    model_state_dict = torch.load(cp_path)['model_state_dict']
    model_state_dict['_actor_logstd'] = model_state_dict['_actor_logstd'].reshape(1, -1)
    agent.load_state_dict(model_state_dict)

    obs = env.reset()
    rollout = [env.unwrapped._state]
    total_reward = 0
    done = False
    while not done:
        with torch.no_grad():
            obs = obs.unsqueeze(dim=0).to(device)
            act, _, _ = agent.get_action(obs)
            act = act.squeeze()
            obs, rew, done, info = env.step(act.cpu())
            print(rew)
            rollout.append(env.unwrapped._state)
            total_reward += rew

    i = Image(image.render(env.unwrapped._env.sys, [s.qp for s in rollout], width=640, height=480))
    display(i)
    print(f'{total_reward=}')
    print(f' Rollout length: {len(rollout)}')


if __name__ == '__main__':
    enjoy()
