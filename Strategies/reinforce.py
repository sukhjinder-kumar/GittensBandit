import numpy as np
from Strategies.strategy_interface import StrategyInterface
import matplotlib.pyplot as plt

class Reinforce(StrategyInterface):
    def __init__(self, 
                 num_arms, 
                 num_states_per_arm,
                 homogeneous,
                 discount_factor, 
                 episode_len, 
                 learning_rate, 
                 temperature):
        super().__init__("Reinforce")
        self.k = num_arms
        self.n = num_states_per_arm
        self.discount_factor = discount_factor
        self.episode_len = episode_len
        self.learning_rate = learning_rate
        self.t = temperature  # Temperature
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
            total = sum([np.exp(self.h[cur_state[i]]/self.t) for i in range(self.k)])
            for i in range(self.k):
                action_probability[i] = np.exp(self.h[cur_state[i]]/self.t) / total
        else:
            total = sum([np.exp(self.h[i][cur_state[i]]/self.t) for i in range(self.k)])
            for i in range(self.k):
                action_probability[i] = np.exp(self.h[i][cur_state[i]]/self.t) / total

        action = int(np.random.choice(range(self.k), p=action_probability))

        return action, action_probability

    def update(self, 
               cur_state,
               action_taken,
               action_probability,
               reward, 
               cumm_reward,
               cur_time):
        
        if self.homogeneous:
            for i in range(self.k):
                if i == action_taken:
                    self.h[cur_state[i]] += (self.learning_rate 
                                             * self.discount_factor**cur_time
                                             * cumm_reward 
                                             * 1/self.t
                                             * (1 - action_probability[i]))
                else:
                    self.h[cur_state[i]] += (self.learning_rate 
                                             * self.discount_factor**cur_time
                                             * cumm_reward
                                             * 1/self.t
                                             * - action_probability[i])
        else:
            for i in range(self.k):
                if i == action_taken:
                    self.h[i][cur_state[i]] += (self.learning_rate 
                                                * self.discount_factor**cur_time
                                                * cumm_reward 
                                                * 1/self.t
                                                * (1 - action_probability[i]))
                else:
                    self.h[i][cur_state[i]] += + (self.learning_rate 
                                                  * self.discount_factor**cur_time
                                                  * cumm_reward 
                                                  * 1/self.t
                                                  * - action_probability[i])
        return

    def reset(self):
        if self.homogeneous:
            self.h = np.zeros((self.n))  # preference for each state
        else:
            self.h = np.zeros((self.k, self.n))  # preference for each (k, n), i.e. kth arm and nth state

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
