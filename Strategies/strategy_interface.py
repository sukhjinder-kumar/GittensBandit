# Interface for all other Strategies

import numpy as np

class StrategyInterface():
    def __init__(self, name):
        self.name = name

    def get_action(self, cur_state):
        # cur_state : np.array((k)) : ith element tells state of ith arm
        action = 0
        action_probability = np.zeros((cur_state.shape[0]))  # okay ..
        action_probability[action] = 1
        return action, action_probability  # \in [0, 1, ... , k-1] which arm to select

    def update_strategy(self, cur_state, action_taken, reward, cummulative_reward=None):
        # action_taken is the output of get_action(cur_state)
        # reward is the reward recieved when we pull action_taken arm
        # cummulative_reward is the discount cummulative reward till the end of episode
        return
