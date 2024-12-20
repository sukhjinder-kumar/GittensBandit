import numpy as np
from numpy import ndarray
from typing import Union
import matplotlib.pyplot as plt

from Strategies.strategy_interface import StrategyInterface

class Reinforce(StrategyInterface):
    def __init__(self, 
                 num_arms, 
                 num_states_per_arm,
                 homogeneous,
                 discount_factor, 
                 episode_len, 
                 learning_rate, 
                 temperature,
                 schedule="none"):
        super().__init__("Reinforce")
        self.k = num_arms
        self.n = num_states_per_arm
        self.discount_factor = discount_factor
        self.episode_len = episode_len
        self.learning_rate = learning_rate

        self.schedule = schedule
        if self.schedule == "none":
            self.cur_temp = temperature
        elif self.schedule == "linear":
            self.max_temp = temperature  # Boltzmann temperature
            self.min_temp = 0.5  # based on Barto et al 1991 (Appendix B)
            self.cur_temp = self.max_temp  # cur_temp
            self.beta = 0.992  # used to update cur_temp

        self.homogeneous = homogeneous
        if self.homogeneous:
            self.h = np.zeros((self.n))  # preference for each state
        else:
            self.h = np.zeros((self.k, self.n))  # preference for each (k, n), i.e. kth arm and nth state

    def get_action(self, cur_state):
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

    def update(self, 
              cur_state: ndarray, 
              next_state: Union[int, None],
              reward: float, 
              action_taken: int, 
              action_probability: Union[ndarray, None],
              cumm_reward: Union[float, None],
              cur_time: Union[int, None]) -> None:
        
        if action_probability is None:
            raise Exception(f"{self.name} recieved wrong param, action_probability, in update method. Can't be None")
        if cumm_reward is None:
            raise Exception(f"{self.name} recieved wrong param, cumm_reward, in update method. Can't be None")
        if cur_time is None:
            raise Exception(f"{self.name} recieved wrong param, cur_time, in update method. Can't be None")

        if self.homogeneous:
            for i in range(self.k):
                if i == action_taken:
                    self.h[cur_state[i]] += (self.learning_rate 
                                             * self.discount_factor**cur_time
                                             * cumm_reward 
                                             * 1/self.cur_temp
                                             * (1 - action_probability[i]))
                else:
                    self.h[cur_state[i]] += (self.learning_rate 
                                             * self.discount_factor**cur_time
                                             * cumm_reward
                                             * 1/self.cur_temp
                                             * - action_probability[i])
        else:
            for i in range(self.k):
                if i == action_taken:
                    self.h[i][cur_state[i]] += (self.learning_rate 
                                                * self.discount_factor**cur_time
                                                * cumm_reward 
                                                * 1/self.cur_temp
                                                * (1 - action_probability[i]))
                else:
                    self.h[i][cur_state[i]] += + (self.learning_rate 
                                                  * self.discount_factor**cur_time
                                                  * cumm_reward 
                                                  * 1/self.cur_temp
                                                  * - action_probability[i])
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
        if self.homogeneous:
            # Plot all n variables on the same figure
            for i in range(self.n):
                plt.plot(h_average[:, i], label=f'h[{i}]')
                plt.text(h_average.shape[0]-1, h_average[:, i][-1], f'({i})', 
                         fontsize=8, verticalalignment='bottom', horizontalalignment='left')
            plt.legend()
            plt.xlabel("episode")
            plt.ylabel("preference")
            plt.title(title)
            plt.savefig(savepath)
            plt.show()
        else:
            # Plot all k x n variables on the same figure
            for i in range(self.k):
                for j in range(self.n):
                    plt.plot(h_average[:, i, j], label=f'h[{i}][{j}]')
                    plt.text(h_average.shape[0]-1, h_average[:, i, j][-1], f'({i},{j})', 
                             fontsize=8, verticalalignment='bottom', horizontalalignment='left')
            plt.legend()
            plt.xlabel("episode")
            plt.ylabel("preference")
            plt.title(title)
            plt.savefig(savepath)
            plt.show()
