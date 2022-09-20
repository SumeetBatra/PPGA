import sys
import gym
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from ribs.archives import GridArchive
from ribs.emitters import ImprovementEmitter
from ribs.optimizers import Optimizer
from ribs.visualize import grid_archive_heatmap


def simulate(env, model, seed=None):
    """Simulates the lunar lander model.

    Args:
        env (gym.Env): A copy of the lunar lander environment.
        model (np.ndarray): The array of weights for the linear policy.
        seed (int): The seed for the environment.
    Returns:
        total_reward (float): The reward accrued by the lander throughout its
            trajectory.
        impact_x_pos (float): The x position of the lander when it touches the
            ground for the first time.
        impact_y_vel (float): The y velocity of the lander when it touches the
            ground for the first time.
    """
    if seed is None:
        seed = 0
    env.reset(seed=seed)

    action_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    model = model.reshape((action_dim, obs_dim))

    total_reward = 0.0
    impact_x_pos = None
    impact_y_vel = None
    all_y_vels = []
    obs = env.reset()[0]
    done = False

    while not done:
        action = np.argmax(model @ obs)  # Linear policy.
        obs, reward, done, info = env.step(action)
        total_reward += reward

        # Refer to the definition of state here:
        # https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py#L306
        x_pos = obs[0]
        y_vel = obs[3]
        leg0_touch = bool(obs[6])
        leg1_touch = bool(obs[7])
        all_y_vels.append(y_vel)

        # Check if the lunar lander is impacting for the first time.
        if impact_x_pos is None and (leg0_touch or leg1_touch):
            impact_x_pos = x_pos
            impact_y_vel = y_vel

    # If the lunar lander did not land, set the x-pos to the one from the final
    # timestep, and set the y-vel to the max y-vel (we use min since the lander
    # goes down).
    if impact_x_pos is None:
        impact_x_pos = x_pos
        impact_y_vel = min(all_y_vels)

    return total_reward, impact_x_pos, impact_y_vel


def train(env, optimizer, seed, archive):
    start_time = time.time()
    total_itrs = 500

    for itr in tqdm(range(1, total_itrs + 1)):
        # Request models from the optimizer.
        sols = optimizer.ask()

        # Evaluate the models and record the objectives and BCs.
        objs, bcs = [], []
        for model in sols:
            obj, impact_x_pos, impact_y_vel = simulate(env, model, seed)
            objs.append(obj)
            bcs.append([impact_x_pos, impact_y_vel])

        # Send the results back to the optimizer.
        optimizer.tell(objs, bcs)

        # Logging.
        if itr % 25 == 0:
            elapsed_time = time.time() - start_time
            print(f"> {itr} itrs completed after {elapsed_time:.2f} s")
            print(f"  - Archive Size: {len(archive)}")
            print(f"  - Max Score: {archive.stats.obj_max}")


if __name__ == '__main__':
    env = gym.make("LunarLander-v2")
    seed = 1339
    action_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]

    archive = GridArchive(
        [50, 50],  # 50 bins in each dimension.
        [(-1.0, 1.0), (-3.0, 0.0)],  # (-1, 1) for x-pos and (-3, 0) for y-vel.
    )

    initial_model = np.zeros((action_dim, obs_dim))
    emitters = [
        ImprovementEmitter(
            archive,
            initial_model.flatten(),
            1.0,  # Initial step size.
            batch_size=30,
        ) for _ in range(5)  # Create 5 separate emitters.
    ]

    optimizer = Optimizer(archive, emitters)

    train(env, optimizer, seed, archive)
    sys.exit(1)
