# Interface for all other Strategies

import numpy as np
from numpy import ndarray
from typing import Annotated


class StrategyInterface():
    def __init__(self, name: str):
        # Overload other parameters
        self.name = name

    def get_action(self, cur_state: Annotated[ndarray, int]) -> tuple[int, Annotated[ndarray, float]]:
        # cur_state : np.array((k)) : ith element tells state of ith arm
        action = int(0)
        action_probability = np.zeros((cur_state.shape[0]))  # Just making stuff up
        action_probability[action] = 1.0
        return action, action_probability  # \in [0, 1, ... , k-1] which arm to select

    def short_term_update(self,
                          cur_state: Annotated[ndarray, int],
                          next_state: int,
                          reward: float,
                          action_taken: int,
                          action_probability: Annotated[ndarray, float],
                          cur_time: int) -> None:
        # action_taken is the output of get_action(cur_state)
        # reward is the reward recieved when we pull action_taken arm
        # cummulative_reward is the discount cummulative reward till the end of episode
        # cur_time is the current iterate in the episode. Algorithms like reinforce require it in discount**t
        # next_state is the new state of selected arm
        pass

    def long_term_update(self,
                         state_history: Annotated[ndarray, int],
                         next_state_history: Annotated[ndarray, int],
                         reward_history: Annotated[ndarray, float], 
                         action_taken_history: Annotated[ndarray, int], 
                         action_probability_history: Annotated[ndarray, float],
                         total_time: int) -> None:
        pass
    
    def reset(self) -> None:
        # reset state variables
        return
