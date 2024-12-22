import numpy as np
from numpy import ndarray
from typing import Union, Annotated

from Strategies.strategy_interface import StrategyInterface
import matplotlib.pyplot as plt


class QLearning(StrategyInterface):
    def __init__(self, 
                 num_arms,
                 num_states_per_arm, 
                 discount_factor, 
                 init_learning_rate, 
                 tau,
                 schedule,
                 max_temperature=None,
                 min_temperature=None,
                 beta=None,
                 epsilon_greedy=None):

        super().__init__("QLearning")

        self.k = num_arms  # Number of arms (tasks)
        self.n = num_states_per_arm  # Number of states per task
        self.discount_factor = discount_factor
        
        # Initialize the Q-table: (cur_state, action, restart_state, task)
        self.q_table = np.zeros((self.n, 2, self.n, self.k))  # Actions: 0 = Continue (C), 1 = Restart (R)
        self.num_updates_q_table = np.zeros((self.n, 2, self.n, self.k))  # stores how many updates for each q value have been performed

        # adaptive learning rate based on Barto et al 1991 (Appendix B)
        self.init_learning_rate = init_learning_rate  
        self.cur_learning_rate = np.full((self.n, 2, self.n, self.k), self.init_learning_rate)  # lr for each q value in the Q table
        self.tau = tau  # used in updating cur_learning_rate (default = 300)
        
        self.schedule = schedule
        if schedule == "Boltzmann":
            if any(x is None for x in (max_temperature, min_temperature, beta)):
                raise Exception("For Boltzmann schedule, temperature can't be None")
            self.max_temp = max_temperature  # Boltzmann temperature (default = 200)
            self.min_temp = min_temperature  # based on Barto et al 1991 (Appendix B, default = 0.5)
            self.cur_temp = self.max_temp
            self.beta = beta  # used to update cur_temp (default = 0.992)
        elif schedule == "epsilon-greedy":
            if epsilon_greedy is None:
                raise Exception("For epsilon-greedy schedule, epsilon can't be None")
            self.epsilon_greedy = epsilon_greedy  # probability of picking action at random

    def get_action(self, cur_state: Annotated[ndarray, int]) -> tuple[int, Annotated[ndarray, float]]:
        action_probabilities = np.zeros((self.k))

        if self.schedule == "Boltzmann":
            # Calculate the Boltzmann distribution for action selection based on current estimate of gittins index
            action_probabilities = np.zeros((self.k))
            total = 0
            for i in range(self.k):
                total += np.exp(self.q_table[cur_state[i], 0, cur_state[i], i] / self.cur_temp)
            for i in range(self.k):
                action_probabilities[i] = np.exp(self.q_table[cur_state[i], 0, cur_state[i], i] / self.cur_temp) / total
                
        elif self.schedule == "epsilon-greedy-gittin":
            # With 1-epsilon probability pick the arm with the state that has highest gitten value
            # ,And with prob epsilon pick randomly
            cur_gittens = np.array([self.q_table[cur_state[i], 0, cur_state[i], i] for i in range(self.k)])
            best_gittens = np.argmax(cur_gittens)
            action_probabilities = np.full(self.k, self.epsilon_greedy / self.k)
            action_probabilities[best_gittens] += (1-self.epsilon_greedy)
            
        # Select action based on the probabilities
        action = int(np.random.choice(range(self.k), p=action_probabilities))
        
        return action, action_probabilities

    def short_term_update(self,
                          cur_state: Annotated[ndarray, int],
                          next_state: int,
                          reward: float,
                          action_taken: int,
                          action_probability: Annotated[ndarray, float],
                          cur_time: int) -> None:

        i = cur_state[action_taken]  # prev_state_of_selected_arm
        j = next_state  # cur_state_of_selected_arm, after transition state
        a = action_taken  # arm selected
        r = reward

        # sorry for the overload of k, here it means any state reached after taking action from i, to i from k.
        # don't confuse with self.k == num_arms, i-j-k seems more natural notation
        
        # update Q table
        for k in range(self.n):
            # Update for Continue (C)
            current_q_value = self.q_table[i, 0, k, a]
            max_next_q_value = max(self.q_table[j, 0, k, a], self.q_table[j, 1, k, a])
            new_q_value = (1 - self.cur_learning_rate[i, 0, k, a]) * current_q_value + \
                            self.cur_learning_rate[i, 0, k, a] * (r + self.discount_factor * max_next_q_value)
            self.q_table[i, 0, k, a] = new_q_value

            # Update for Restart (R)
            current_q_value_restart = self.q_table[k, 1, i, a]
            max_next_q_value_restart = max(self.q_table[j, 0, i, a], self.q_table[j, 1, i, a])
            new_q_value_restart = (1-self.cur_learning_rate[k, 1, i, a]) * current_q_value_restart + \
                                    self.cur_learning_rate[k, 1, i, a] * (r + self.discount_factor * max_next_q_value_restart)
            self.q_table[k, 1, i, a] = new_q_value_restart

        # update num_updates_q_table
        for k in range(self.n):
            self.num_updates_q_table[i, 0, k, a] += 1
            self.num_updates_q_table[k, 1, i, a] += 1

        # update learning rate
        for k in range(self.n):
            self.cur_learning_rate[i, 0, k, a] = (self.init_learning_rate * self.tau) / (self.tau + self.num_updates_q_table[i, 0, k, a]) 
            self.cur_learning_rate[k, 1, i, a] = (self.init_learning_rate * self.tau) / (self.tau + self.num_updates_q_table[k, 1, i, a]) 

        # decrease the temperature
        if self.schedule == "Boltzmann":
            self.cur_temp = self.min_temp + self.beta * (self.cur_temp - self.min_temp)


    def long_term_update(self,
                         state_history: Annotated[ndarray, int],
                         next_state_history: Annotated[ndarray, int],
                         reward_history: Annotated[ndarray, float], 
                         action_taken_history: Annotated[ndarray, int], 
                         action_probability_history: Annotated[ndarray, float],
                         total_time: int) -> None:
        pass

    def reset(self): 
        self.q_table = np.zeros((self.n, 2, self.n, self.k))
        self.cur_learning_rate = np.full((self.n, 2, self.n, self.k), self.init_learning_rate)
        self.num_updates_q_table = np.zeros((self.n, 2, self.n, self.k))
        if self.schedule == "Boltzmann":
            self.cur_temp = self.max_temp
    
    def qlearning_visualize(self, gittin_history, title, save_path):
        num_runs, time_steps, num_arms, num_states_per_arm = gittin_history.shape
        
        for k in range(num_arms):
            for n in range(num_states_per_arm):
                values = np.mean(gittin_history[:, :, k, n], axis=0)
                plt.plot(values, label=f'({k},{n})')
                
                # Add index (k, n) as a text label above the line at the last time step
                plt.text(time_steps - 1, values[-1], f'({k},{n})', 
                         fontsize=8, verticalalignment='bottom', horizontalalignment='left')
                
        # Add labels and title
        plt.xlabel('Time')
        plt.ylabel('Gittins Index')
        plt.title(title)
        
        # Show legend
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small', title='(k,n) Pairs')
        
        # Save fig
        plt.savefig(save_path)

        # Display the plot
        plt.show()
