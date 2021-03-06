import enum
from random import randint

import gym
from gym import spaces


class TrainSimEnv(gym.Env):
    metadata = {'render.modes': ['console']}
    rewards = {'tiny': 0.1, 'small': 1, 'medium': 5, 'big': 10}

    def __init__(self):
        """
        # Action space
        # the following actions can be taken:
        # 0: wait/do nothing
        # 1: get passenger
        # 2: put passenger
        # 3: move left (lower numbers)
        # 4: move right (higher numbers)

         # Observation space
        # stations:
        # (1) - (2) - (3) - (4) - (5)
        # each station can have one of the following states:
        # 0: no passenger
        # 1: passenger bound for 1
        # 2: passenger bound for 2
        # 3: passenger bound for 3
        # 4: passenger bound for 4
        # 5: passenger bound for 5
        #
        # train:
        # moves one station in a timestep
        # train position: 1, 2, 3, 4, 5
        # train can have the following states:
        # --> see station states
        #
        """
        super(TrainSimEnv, self).__init__()

        self.action_space = spaces.Discrete(5)

        self.number_of_stations = 5
        train_states = 6
        station_states = 6
        self.observation_space = spaces.MultiDiscrete([
            self.number_of_stations,  # possible train positions
            train_states,  # train states (passengers)
            station_states,  # state of station 1
            station_states,  # state of station 2
            station_states,  # state of station 3
            station_states,  # state of station 4
            station_states,  # state of station 5
        ])

        self.state = [0, 0, 0, 0, 0, 0, 0]

        # timestep counter
        self.time = 0
        self.max_time = 1000

        # passenger occurs randomly after n timesteps
        self.max_time_passenger = 10
        self.time_until_next_passenger = 0  # starts with a passenger waiting
        self.max_passengers = 50

        # for console output
        self.overall_passengers = 0
        self.to_correct_station = 0
        self.to_wrong_station = 0
        self.skipped_waiting = 0

    def step(self, action):
        self.reward = 0
        self.done = False

        # logic for creating new passenger
        if self.time_until_next_passenger == 0:
            station = randint(1, 5)
            destination = randint(1, 5)
            while destination == station:
                destination = randint(1, 5)
            # if there already is a passenger, then this passenger has abandoned waiting
            # this results in a small loss (unhappy passenger)
            if self.state[station + 1] > 0:
                #self.reward = self.reward - self.rewards['medium']
                self.skipped_waiting = self.skipped_waiting + 1
            self.state[station + 1] = destination
            self.time_until_next_passenger = self.max_time_passenger
            self.overall_passengers = self.overall_passengers + 1
        else:
            self.time_until_next_passenger = self.time_until_next_passenger - 1

        # work actions
        current_position = self.state[0] + 1
        current_state = self.state[1]
        station_state = self.state[current_position + 1]
        if action == int(Actions.wait):
            pass
        elif action == int(Actions.get):
            if station_state == 0:
                #self.reward = self.reward - self.rewards['small']
                pass
            else:
                if current_state == 0:
                    self.state[1] = station_state
                    self.state[current_position + 1] = 0
                    #self.reward = self.reward + self.rewards['medium']
                else:
                    #self.reward = self.reward - self.rewards['small']
                    pass
        elif action == int(Actions.put):
            if current_state == 0:
                #self.reward = self.reward - self.rewards['small']
                pass
            else:
                self.state[1] = 0
                if current_position == current_state:
                    self.reward = self.reward + self.rewards['big']
                    self.to_correct_station = self.to_correct_station + 1
                else:
                    #self.reward = self.reward - self.rewards['medium']
                    self.to_wrong_station = self.to_wrong_station + 1
                    self.done = True
        elif action == int(Actions.left):
            if current_position > 1:
                self.state[0] = self.state[0] - 1
                if self.state[1] > 0:
                    #self.reward = self.reward - self.rewards['small']
                    pass
        elif action == int(Actions.right):
            if current_position < 5:
                self.state[0] = self.state[0] + 1
                if self.state[1] > 0:
                    #self.reward = self.reward - self.rewards['small']
                    pass

        # bookkeeping
        self.time = self.time + 1
        self.reward = self.reward - self.rewards['tiny']
        if self.time > self.max_time:
            self.done = True
        if self.to_wrong_station + self.to_correct_station + self.skipped_waiting == self.max_passengers:
            self.done = True

        info = {}

        return self.state, self.reward, self.done, info

    def reset(self):
        self.state = [0, 0, 0, 0, 0, 0, 0]
        self.time_until_next_passenger = 0
        self.time = 0
        self.overall_passengers = 0
        self.to_correct_station = 0
        self.to_wrong_station = 0
        self.skipped_waiting = 0
        return self.state

    def render(self, mode="console"):
        print('#' + str(self.time) + ': (' +
              str(self.overall_passengers) + '/' +
              str(self.to_correct_station) + '/' +
              str(self.to_wrong_station) + '/' +
              str(self.skipped_waiting) + ')')

    def close(self):
        pass


class Actions(enum.IntEnum):
    wait = 0
    get = 1
    put = 2
    left = 3
    right = 4
