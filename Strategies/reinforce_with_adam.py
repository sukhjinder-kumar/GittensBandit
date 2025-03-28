import numpy as np
from numpy import ndarray
from typing import Union, Annotated
import matplotlib.pyplot as plt

from Strategies.strategy_interface import StrategyInterface

class ReinforceWithAdam(StrategyInterface):
    def __init__(self, 
                 num_arms, 
                 num_states_per_arm,
                 homogeneous,
                 discount_factor, 
                 learning_rate=0.001, 
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 schedule='none',
                 constant_temperature=None,
                 max_temperature=None,
                 min_temperature=None,
                 beta=None,
                 name="ReinforceWithAdam"):
        super().__init__(name)
        self.k = num_arms
        self.n = num_states_per_arm
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        
        # Adam optimization hyperparameters
        self.beta1 = beta1  # Decay rate for first moment estimate
        self.beta2 = beta2  # Decay rate for second moment estimate
        self.epsilon = epsilon  # Small value to prevent division by zero

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

        # Preference, first moment, and second moment initialization
        self.homogeneous = homogeneous
        if self.homogeneous:
            self.h = np.zeros((self.n))
            self.m = np.zeros((self.n))  # First moment vector
            self.v = np.zeros((self.n))  # Second moment vector
        else:
            self.h = np.zeros((self.k, self.n))
            self.m = np.zeros((self.k, self.n))  # First moment vector
            self.v = np.zeros((self.k, self.n))  # Second moment vector
        
        # Iteration counter for bias correction
        self.t = 0

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

        # Increment iteration counter
        self.t += 1

        # Adam optimization update
        for t in range(total_time):
            if self.homogeneous:
                for i in range(self.k):
                    # Compute gradient
                    gradient = (self.learning_rate 
                                * self.discount_factor**t
                                * cumm_reward_history[t] 
                                * 1/self.cur_temp
                                * (indicator_func(i, action_taken_history[t]) - action_probability_history[t][i]))
                    
                    # Update first moment estimate (momentum)
                    self.m[state_history[t][i]] = (
                        self.beta1 * self.m[state_history[t][i]] + 
                        (1 - self.beta1) * gradient
                    )
                    
                    # Update second moment estimate (RMSprop-like)
                    self.v[state_history[t][i]] = (
                        self.beta2 * self.v[state_history[t][i]] + 
                        (1 - self.beta2) * (gradient ** 2)
                    )
                    
                    # Bias correction
                    m_hat = self.m[state_history[t][i]] / (1 - self.beta1 ** self.t)
                    v_hat = self.v[state_history[t][i]] / (1 - self.beta2 ** self.t)
                    
                    # Update preferences using Adam update rule
                    self.h[state_history[t][i]] += (
                        m_hat / (np.sqrt(v_hat) + self.epsilon)
                    )
            else:
                for i in range(self.k):
                    # Compute gradient
                    gradient = (self.learning_rate 
                                * self.discount_factor**t
                                * cumm_reward_history[t]
                                * 1/self.cur_temp
                                * (indicator_func(i, action_taken_history[t]) - action_probability_history[t][i]))
                    
                    # Update first moment estimate (momentum)
                    self.m[i][state_history[t][i]] = (
                        self.beta1 * self.m[i][state_history[t][i]] + 
                        (1 - self.beta1) * gradient
                    )
                    
                    # Update second moment estimate (RMSprop-like)
                    self.v[i][state_history[t][i]] = (
                        self.beta2 * self.v[i][state_history[t][i]] + 
                        (1 - self.beta2) * (gradient ** 2)
                    )
                    
                    # Bias correction
                    m_hat = self.m[i][state_history[t][i]] / (1 - self.beta1 ** self.t)
                    v_hat = self.v[i][state_history[t][i]] / (1 - self.beta2 ** self.t)
                    
                    # Update preferences using Adam update rule
                    self.h[i][state_history[t][i]] += (
                        m_hat / (np.sqrt(v_hat) + self.epsilon)
                    )

            # Temperature scheduling
            if self.schedule == "linear":
                self.cur_temp = self.min_temp + self.beta * (self.cur_temp - self.min_temp)

    def reset(self):
        # Reset preferences, first and second moment vectors
        if self.homogeneous:
            self.h = np.zeros((self.n))
            self.m = np.zeros((self.n))
            self.v = np.zeros((self.n))
        else:
            self.h = np.zeros((self.k, self.n))
            self.m = np.zeros((self.k, self.n))
            self.v = np.zeros((self.k, self.n))

        # Reset iteration counter and temperature
        self.t = 0
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