import numpy as np
from numpy import ndarray
from typing import Union, Annotated
import matplotlib.pyplot as plt

from Strategies.strategy_interface import StrategyInterface

class Reinforce(StrategyInterface):
    def __init__(self, 
                 num_arms, 
                 num_states_per_arm,
                 homogeneous,
                 discount_factor, 
                 learning_rate, 
                 schedule,
                 constant_temperature=None,
                 max_temperature=None,
                 min_temperature=None,
                 beta=None):
        super().__init__("Reinforce")
        self.k = num_arms
        self.n = num_states_per_arm
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

        self.schedule = schedule
        if self.schedule == "none":
            if constant_temperature is None:
                raise Exception("For none schedule, constant_temperature can't be None")
            self.cur_temp = constant_temperature
        elif self.schedule == "linear":
            if any(x is None for x in (max_temperature, min_temperature, beta)):
                raise Exception("For linear schedule, temperature can't be None")
            self.max_temp = max_temperature  # Boltzmann temperature
            self.min_temp = min_temperature  # based on Barto et al 1991 (Appendix B)
            self.cur_temp = self.max_temp  # cur_temp
            self.beta = beta  # used to update cur_temp

        self.homogeneous = homogeneous
        if self.homogeneous:
            self.h = np.zeros((self.n))  # preference for each state
        else:
            self.h = np.zeros((self.k, self.n))  # preference for each (k, n), i.e. kth arm and nth state

    def get_action(self, cur_state: Annotated[ndarray, int]) -> tuple[int, Annotated[ndarray, float]]:

        '''
        Using Boltzmann Sampling
        cur_states : np.array((self.k)) : state of each arm
        '''
        action_probability = np.zeros((self.k))  # probability of selecting each arm
        
        if self.homogeneous:
            total = sum([np.exp(self.h[cur_state[i]]/self.cur_temp) for i in range(self.k)])
            for i in range(self.k):
                action_probability[i] = np.exp(self.h[cur_state[i]]/self.cur_temp) / total
        else:
            total = sum([np.exp(self.h[i][cur_state[i]]/self.cur_temp) for i in range(self.k)])
            for i in range(self.k):
                action_probability[i] = np.exp(self.h[i][cur_state[i]]/self.cur_temp) / total

        action = int(np.random.choice(range(self.k), p=action_probability))

        return action, action_probability

    def short_term_update(self,
                          cur_state: Annotated[ndarray, int],
                          next_state: int,
                          reward: float,
                          action_taken: int,
                          action_probability: Annotated[ndarray, float],
                          cur_time: int) -> None:
        pass

    def long_term_update(self,
                         state_history: Annotated[ndarray, int],
                         next_state_history: Annotated[ndarray, int],
                         reward_history: Annotated[ndarray, float], 
                         action_taken_history: Annotated[ndarray, int], 
                         action_probability_history: Annotated[ndarray, float],
                         total_time: int) -> None:
        
        # track cummulative reward
        cumm_reward_history = np.zeros((total_time), dtype=float)
        counter = 0
        for t in range(total_time):
            counter = reward_history[t] + self.discount_factor * counter
            cumm_reward_history[t] = counter

        def indicator_func(a, b):
            if a==b:
                return 1
            else:
                return 0

        for t in range(total_time):
            if self.homogeneous:
                for i in range(self.k):
                    self.h[state_history[t][i]] += (self.learning_rate 
                                                    * self.discount_factor**t
                                                    * cumm_reward_history[t] 
                                                    * 1/self.cur_temp
                                                    * (indicator_func(i, action_taken_history[t]) - action_probability_history[t][i]))
            else:
                for i in range(self.k):
                    self.h[i][state_history[t][i]] += (self.learning_rate 
                                                       * self.discount_factor**t
                                                       * cumm_reward_history[t]
                                                       * 1/self.cur_temp
                                                       * (indicator_func(i, action_taken_history[t]) - action_probability_history[t][i]))

            # decrease the temp if schedule is not None
            if self.schedule == "linear":
                self.cur_temp = self.min_temp + self.beta * (self.cur_temp - self.min_temp)

        return

    def reset(self):
        if self.homogeneous:
            self.h = np.zeros((self.n))  # preference for each state
        else:
            self.h = np.zeros((self.k, self.n))  # preference for each (k, n), i.e. kth arm and nth state

        if self.schedule == "linear":
            self.cur_temp = self.max_temp

    def visualize_h_average(self, h_average, title, savepath):
        plt.figure(figsize=(10, 6))  # Set figure size

        if self.homogeneous:
            # Plot all n variables on the same figure
            for i in range(self.n):
                plt.plot(h_average[:, i], label=f'h[{i}]')
                plt.text(h_average.shape[0]-1, h_average[:, i][-1], f'({i})', 
                         fontsize=8, verticalalignment='bottom', horizontalalignment='left')
        else:
            # Plot all k x n variables on the same figure
            for i in range(self.k):
                for j in range(self.n):
                    plt.plot(h_average[:, i, j], label=f'h[{i}][{j}]')
                    plt.text(h_average.shape[0]-1, h_average[:, i, j][-1], f'({i},{j})', 
                             fontsize=8, verticalalignment='bottom', horizontalalignment='left')

        # Position the legend outside the plot
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small', title='h[i]')
    
        plt.xlabel("episode")
        plt.ylabel("preference")
        plt.title(title)

        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig(savepath, bbox_inches='tight')
        plt.show()