import sys
import gym
import QDgym
import numpy as np


def qd_gym():
    env = gym.make("QDHalfCheetahBulletEnv-v0", render=True)
    state = env.reset()
    done = False
    for _ in range(2000):
        print(f'{done=}')
        rand_act = np.random.randn(8)
        obs, rew, done, _ = env.step(rand_act)
    fitness = env.tot_reward
    bc = env.desc
    env.close()


if __name__ == '__main__':
    qd_gym()
    sys.exit(1)
