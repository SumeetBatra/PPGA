import gym


class ForwardReward(gym.core.RewardWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.prev_x_pos = self.env.robot.body_xyz[0]

    def reward(self, reward):
        cur_x_pos = self.env.robot.body_xyz[0]
        dx = cur_x_pos - self.prev_x_pos
        self.prev_x_pos = cur_x_pos
        return reward + abs(dx)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        rew = self.reward(rew)
        return obs, rew, done, info


class TotalReward(gym.core.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.total_reward = 0

    def reward(self, reward):
        pass

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.total_reward += rew
        info['total_reward'] = self.total_reward
        return obs, rew, done, info

    def reset(self):
        self.total_reward = 0
        obs = self.env.reset()
        return obs
