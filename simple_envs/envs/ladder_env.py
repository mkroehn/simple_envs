import gym
from gym import spaces


class LadderEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self):
        self.action_space = spaces.Discrete(2)  # allow up or down
        self.observation_space = spaces.Discrete(10)  # ladder with 10 steps
        self.state = 1  # first step of the ladder

    def step(self, action):
        done = False
        reward = -1  # small negative reward for every step taken
        if action == 0:
            self.state = self.state - 1  # climbing down
            if self.state < 1:
                self.state = 1
        if action == 1:
            self.state = self.state + 1  # climbing down
            if self.state == 10:
                reward = reward + 100  # big positive reward when climed all up
                done = True
        info = {}
        return self.state, reward, done, info

    def reset(self):
        self.state = 1
        return self.state

    def render(self, mode='console'):
        print(self.state)

    def close(self):
        pass
