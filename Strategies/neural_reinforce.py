import numpy as np
import torch
import torch.nn as nn
from torch.optim.adam import Adam

from Strategies.strategy_interface import StrategyInterface
import matplotlib.pyplot as plt


class PreferenceNonHomogeneousNN(nn.Module):
    '''
    Input R^2 : (k, n) : kth arm and nth state
    Output R^1 : preference or h[k][n]

    Usage :
    h = PreferenceNonHomogeneousNN()
    h(1,2)  # h[1][2] : returns tensor, do .item() for just the value.
    '''
    def __init__(self):
        super(PreferenceNonHomogeneousNN, self).__init__()
        self.layer1 = nn.Linear(2,4)
        self.layer2 = nn.Linear(4,2)
        self.layer3 = nn.Linear(2,1)
        self.R = nn.ReLU()
    def forward(self, k, n):
        x = torch.tensor([k, n], dtype=torch.float)
        x = self.R(self.layer1(x))
        x = self.R(self.layer2(x))
        x = self.layer3(x)
        return x.squeeze()


class PreferenceHomogeneousNN(nn.Module):
    '''
    Input R : (n) : nth state
    Output R : preference or h[n]

    Usage :
    h = PreferenceHomogeneousNN()
    h(1)  # h[1] : returns tensor, do .item() for just the value.
    '''
    def __init__(self):
        super(PreferenceHomogeneousNN, self).__init__()
        self.layer1 = nn.Linear(1,2)
        self.layer2 = nn.Linear(2,2)
        self.layer3 = nn.Linear(2,1)
        self.R = nn.ReLU()
    def forward(self, n):
        x = torch.tensor([n], dtype=torch.float)
        x = self.R(self.layer1(x))
        x = self.R(self.layer2(x))
        x = self.layer3(x)
        return x.squeeze()


class NeuralReinforce(StrategyInterface):
    def __init__(self, 
                 num_arms,
                 num_states_per_arm, 
                 homogeneous,
                 discount_factor, 
                 episode_len, 
                 learning_rate,
                 max_temperature, 
                 schedule):
        super().__init__("NeuralReinforce")
        self.k = num_arms
        self.n = num_states_per_arm
        self.discount_factor = discount_factor
        self.episode_len = episode_len
        self.learning_rate = learning_rate
        
        self.schedule = schedule  # implemented = ["linear"]
        if self.schedule == "linear":
            self.min_t = 0.5  # temperature
            self.max_t = max_temperature
            self.cur_t = self.max_t
            self.beta = 0.992

        self.homogeneous = homogeneous
        if self.homogeneous:
            self.h = PreferenceHomogeneousNN()  # preference for each state
        else:
            self.h = PreferenceNonHomogeneousNN()   # preference for each (k, n), i.e. kth arm and nth state

        self.optimizer = Adam(self.h.parameters(), lr=self.learning_rate)

    def get_action(self, cur_state):
        '''
        Using Boltzmann Sampling
        cur_states : state of each arm : np.array((k), dtype=int)
        '''
        action_probability = np.zeros((self.k))  # probability of selecting each arm
        
        if self.homogeneous:
            total = sum([np.exp(self.h(cur_state[i]).item()/self.cur_t) for i in range(self.k)])
            for i in range(self.k):
                action_probability[i] = np.exp(self.h(cur_state[i]).item()/self.cur_t) / total
        else:
            total = sum([np.exp(self.h(i, cur_state[i]).item()/self.cur_t) for i in range(self.k)])
            for i in range(self.k):
                action_probability[i] = np.exp(self.h(i, cur_state[i]).item()/self.cur_t) / total
                
        action = int(np.random.choice(range(self.k), p=action_probability))
                
        return action, action_probability

    def update(self, 
               cur_state,
               action_taken,
               action_probability,
               reward, 
               cumm_reward,
               cur_time):
        # update preferences
        if self.homogeneous:
            self.optimizer.zero_grad()
            loss = torch.tensor([0], dtype=torch.float)
            for i in range(self.k):
                if i == action_taken:
                    loss += torch.tensor([1 - action_probability[i]], dtype=torch.float) * self.h(cur_state[i])
                else:
                    loss += torch.tensor([- action_probability[i]], dtype=torch.float) * self.h(cur_state[i])
            loss *= -(self.discount_factor**cur_time
                      * cumm_reward
                      * 1/self.cur_t)
            loss.backward()
            self.optimizer.step() 
        else:
            self.optimizer.zero_grad()
            loss = torch.tensor([0], dtype=torch.float)
            for i in range(self.k):
                if i == action_taken:
                    loss += torch.tensor([1 - action_probability[i]], dtype=torch.float) * self.h(i, cur_state[i])
                else:
                    loss += torch.tensor([- action_probability[i]], dtype=torch.float) * self.h(i, cur_state[i])
            loss *= -(self.discount_factor**cur_time
                      * cumm_reward 
                      * 1/self.cur_t)
            loss.backward()
            self.optimizer.step()

        # update temperature
        if self.schedule == "linear":
            self.cur_t = self.min_t + self.beta * (self.cur_t - self.min_t)

        return

    def reset(self):
        if self.homogeneous:
            self.h = PreferenceHomogeneousNN()  # preference for each state
        else:
            self.h = PreferenceNonHomogeneousNN()   # preference for each (k, n), i.e. kth arm and nth state

        # As the model parameter "id" is changed, optimizer needs to be reinitialzed too.
        self.optimizer = Adam(self.h.parameters(), lr=self.learning_rate)
            
        if self.schedule == "linear":
            self.cur_t = self.max_t
            
        return

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
