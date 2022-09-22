import argparse
import sys
from distutils.util import strtobool
from attrdict import AttrDict
from utils.utils import config_wandb
from RL.ppo import PPO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, help="Choose from [QDAntBulletEnv-v0,"
                                                     "QDHalfCheetahBulletEnv-v0]")
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument("--torch_deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--use_wandb", default=False, type=lambda x: bool(strtobool(x)),
                        help='Use weights and biases to track the exp')
    parser.add_argument('--wandb_run_name', type=str, default='ppo_ant')

    # algorithm args
    parser.add_argument('--total_timesteps', type=int, default=1000000)
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker processes to spawn. '
                             'Should always be <= number of logical cores on your machine')
    parser.add_argument('--envs_per_worker', type=int, default=1,
                        help='Num envs each worker process will step through sequentially')
    parser.add_argument('--num_steps', type=int, default=2048,
                        help='the number of steps to run in each environment per policy rollout')
    # ppo hyperparams
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--anneal_lr', type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help='Toggle learning rate annealing for policy and value networks')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for rewards')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='Lambda discount used for general advantage est')
    parser.add_argument('--num_minibatches', type=int, default=32)
    parser.add_argument('--update_epochs', type=int, default=10, help='The K epochs to update the policy')
    parser.add_argument("--norm_adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip_coef", type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip_vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--entropy_coef", type=float, default=0.0,
                        help="coefficient of the entropy")
    parser.add_argument("--vf_coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target_kl", type=float, default=None,
                        help="the target KL divergence threshold")

    args = parser.parse_args()
    args.batch_size = int(args.num_workers * args.envs_per_worker * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    cfg = AttrDict(vars(args))
    return cfg


if __name__ == '__main__':
    cfg = parse_args()

    if cfg.use_wandb:
        config_wandb(batch_size=cfg.batch_size, total_steps=cfg.total_timesteps, run_name=cfg.wandb_run_name)

    alg = PPO(cfg.seed, cfg)
    num_updates = cfg.total_timesteps // cfg.batch_size
    alg.train(num_updates, traj_len=cfg.num_steps)
    sys.exit(0)
