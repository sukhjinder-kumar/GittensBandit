# Interface for all other Strategies

import numpy as np
from numpy import ndarray
from typing import Union

class StrategyInterface():
    def __init__(self, name: str):
        self.name = name

    def get_action(self, cur_state: ndarray) -> tuple[int, ndarray]:
        # cur_state : np.array((k)) : ith element tells state of ith arm
        action = int(0)
        action_probability = np.zeros((cur_state.shape[0]))  # okay ..
        action_probability[action] = 1.0
        return action, action_probability  # \in [0, 1, ... , k-1] which arm to select

    def update(self, 
              cur_state: ndarray, 
              next_state: Union[int, None],
              reward: float, 
              action_taken: int, 
              action_probability: Union[ndarray, None],
              cumm_reward: Union[float, None],
              cur_time: Union[int, None]) -> None:
        # action_taken is the output of get_action(cur_state)
        # reward is the reward recieved when we pull action_taken arm
        # cummulative_reward is the discount cummulative reward till the end of episode
        # cur_time is the current iterate in the episode. Algorithms like reinforce require it in discount**t
        # next_state is the new state of selected arm
        return
    
    def reset(self) -> None:
        # reset state variables
        return
