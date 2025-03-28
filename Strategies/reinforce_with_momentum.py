import numpy as np
from numpy import ndarray
from typing import Union, Annotated
import matplotlib.pyplot as plt

from Strategies.strategy_interface import StrategyInterface

class ReinforceWithMomentum(StrategyInterface):
    def __init__(self, 
                 num_arms, 
                 num_states_per_arm,
                 homogeneous,
                 discount_factor, 
                 learning_rate, 
                 momentum=0.9,
                 schedule='none',
                 constant_temperature=None,
                 max_temperature=None,
                 min_temperature=None,
                 beta=None,
                 name="ReinforceWithMomentum"):
        super().__init__(name)
        self.k = num_arms
        self.n = num_states_per_arm
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.momentum = momentum

        # Temperature scheduling setup
        self.schedule = schedule
        if self.schedule == "none":
            if constant_temperature is None:
                raise Exception("For none schedule, constant_temperature can't be None")
            self.cur_temp = constant_temperature
        elif self.schedule == "linear":
            if any(x is None for x in (max_temperature, min_temperature, beta)):
                raise Exception("For linear schedule, temperature can't be None")
            self.max_temp = max_temperature
            self.min_temp = min_temperature
            self.cur_temp = self.max_temp
            self.beta = beta

        # Preference and momentum initialization
        self.homogeneous = homogeneous
        if self.homogeneous:
            self.h = np.zeros((self.n))
            self.h_momentum = np.zeros((self.n))
        else:
            self.h = np.zeros((self.k, self.n))
            self.h_momentum = np.zeros((self.k, self.n))

    def get_action(self, cur_state: Annotated[ndarray, int]) -> tuple[int, Annotated[ndarray, float]]:
        action_probability = np.zeros((self.k))
        
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

    def long_term_update(self,
                         state_history: Annotated[ndarray, int],
                         next_state_history: Annotated[ndarray, int],
                         reward_history: Annotated[ndarray, float], 
                         action_taken_history: Annotated[ndarray, int], 
                         action_probability_history: Annotated[ndarray, float],
                         total_time: int) -> None:
        
        # Compute cumulative rewards
        cumm_reward_history = np.zeros((total_time), dtype=float)
        counter = 0
        for t in reversed(range(total_time)):
            counter = reward_history[t] + self.discount_factor * counter
            cumm_reward_history[t] = counter

        def indicator_func(a, b):
            return 1 if a == b else 0

        # Momentum-based gradient update
        for t in range(total_time):
            if self.homogeneous:
                for i in range(self.k):
                    # Compute gradient
                    gradient = (self.learning_rate 
                                * self.discount_factor**t
                                * cumm_reward_history[t] 
                                * 1/self.cur_temp
                                * (indicator_func(i, action_taken_history[t]) - action_probability_history[t][i]))
                    
                    # Update momentum
                    self.h_momentum[state_history[t][i]] = (
                        self.momentum * self.h_momentum[state_history[t][i]] + 
                        (1 - self.momentum) * gradient
                    )
                    
                    # Update preferences with momentum
                    self.h[state_history[t][i]] += self.h_momentum[state_history[t][i]]
            else:
                for i in range(self.k):
                    # Compute gradient
                    gradient = (self.learning_rate 
                                * self.discount_factor**t
                                * cumm_reward_history[t]
                                * 1/self.cur_temp
                                * (indicator_func(i, action_taken_history[t]) - action_probability_history[t][i]))
                    
                    # Update momentum
                    self.h_momentum[i][state_history[t][i]] = (
                        self.momentum * self.h_momentum[i][state_history[t][i]] + 
                        (1 - self.momentum) * gradient
                    )
                    
                    # Update preferences with momentum
                    self.h[i][state_history[t][i]] += self.h_momentum[i][state_history[t][i]]

            # Temperature scheduling
            if self.schedule == "linear":
                self.cur_temp = self.min_temp + self.beta * (self.cur_temp - self.min_temp)

    def reset(self):
        # Reset preferences and momentum
        if self.homogeneous:
            self.h = np.zeros((self.n))
            self.h_momentum = np.zeros((self.n))
        else:
            self.h = np.zeros((self.k, self.n))
            self.h_momentum = np.zeros((self.k, self.n))

        # Reset temperature if using linear schedule
        if self.schedule == "linear":
            self.cur_temp = self.max_temp

    def visualize_h_average(self, h_average, title, savepath):
        plt.figure(figsize=(10, 6))

        if self.homogeneous:
            for i in range(self.n):
                plt.plot(h_average[:, i], label=f'h[{i}]')
                plt.text(h_average.shape[0]-1, h_average[:, i][-1], f'({i})', 
                         fontsize=8, verticalalignment='bottom', horizontalalignment='left')
        else:
            for i in range(self.k):
                for j in range(self.n):
                    plt.plot(h_average[:, i, j], label=f'h[{i}][{j}]')
                    plt.text(h_average.shape[0]-1, h_average[:, i, j][-1], f'({i},{j})', 
                             fontsize=8, verticalalignment='bottom', horizontalalignment='left')

        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small', title='h[i]')
        plt.xlabel("episode")
        plt.ylabel("preference")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(savepath, bbox_inches='tight')
        plt.show()