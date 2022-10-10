from attrdict import AttrDict
from RL.ppo import *
from utils.utils import log
from envs.env import make_env
from models.actor_critic import QDActorCriticShared


def enjoy():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # env = gym.make("QDAntBulletEnv-v0", render=True)
    # env.seed(8679)
    env = make_env('QDAntBulletEnv-v0', seed=0, gamma=0.99)()
    env.render()
    obs_shape, action_shape = env.observation_space.shape, env.action_space.shape
    cp_path = "checkpoints/checkpoint0"
    model_state_dict = torch.load(cp_path)['model_state_dict']
    cfg = {'num_workers': 1, 'envs_per_worker': 1, 'normalize_obs': True, 'normalize_rewards': True, 'num_dims': 4, 'envs_per_model': 1}
    cfg = AttrDict(cfg)
    agent = QDActorCriticShared(cfg, obs_shape, action_shape, 4).to(device)
    # agent = QDVectorizedActorCriticShared(cfg, [agent], QDActorCriticShared, measure_dims=4)
    model_state_dict['_actor_logstd'] = model_state_dict['_actor_logstd'].reshape(1, -1)
    # agent.load_state_dict(model_state_dict)
    obs_mean, obs_var = agent.obs_normalizer.obs_rms.mean, agent.obs_normalizer.obs_rms.var
    obs_mean = obs_mean.to(device)
    obs_var = obs_var.to(device)

    obs = env.reset()
    obs = torch.from_numpy(obs).to(device).reshape(1, -1)
    done = False
    total_reward = 0
    while not done:
        # log.debug(f'{done=}')
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
    env.close()


if __name__ == '__main__':
    enjoy()
