from typing import Union
import numpy as np
from numpy import ndarray
from Environment.test_cases import test2
from Environment.mp_environment import MP


class MAB():
    def __init__(self,
                 num_arms, 
                 num_states_per_arm, 
                 transition_matrices, 
                 reward_matrices, 
                 homogeneous,
                 initial_start_state: Union[ndarray, None] = None):
        '''
        If homogeneous == True

            transition_matrices : np.array((num_states_per_arm, num_states_per_arm))
                                : t_m[1][2] -> probability of going from state (0,1) to (0,2) for any arm
            reward_matrices : np.array((num_states_per_arm, num_states_per_arm))
                            : r_m[1][2] -> reward when state transitions from (0,1) to (0,2) for any arm

        If homogeneous == False

            transition_matrices : np.array((num_arms, num_states_per_arm, num_states_per_arm))
                                : t_m[0][1][2] -> probability of going from state (0,1) to (0,2) when machine 0 is pressed
            reward_matrices : np.array((num_arms, num_states_per_arm, num_states_per_arm))
                            : r_m[0][1][2] -> reward when state transitions from (0,1) to (0,2)
                            : Note - here reward is still random, but reward and next state are dependent (100% correlated)
                            : Based on examples covered in Duffs 1995 paper

        initial_start_state : np.array((num_arms)) : initial states of each arm
        '''
        self.k = num_arms  # number of markovian machines
        self.n = num_states_per_arm  # number of states in markovian machine
        self.machines = np.empty(self.k, dtype=object)  # list of MP class objects
        self.homogeneous = homogeneous
        if initial_start_state is None:
            self.initial_start_state = np.random.choice(self.n, self.k) 
        else:
            self.initial_start_state = initial_start_state
        for i in range(self.k):
            if homogeneous:
                machine = MP(num_states = self.n, 
                             transition_matrix = transition_matrices,
                             reward_matrix = reward_matrices,
                             start_state=self.initial_start_state[i])
            else:
                machine = MP(num_states = self.n, 
                             transition_matrix = transition_matrices[i],
                             reward_matrix = reward_matrices[i],
                             start_state=self.initial_start_state[i])
            self.machines[i] = machine

    def step(self, selected_arm):  
        '''
        input: action - which machine in [0, 1, ... , self.K-1] to select
        returns: reward
        '''
        next_machine_state, reward = self.machines[selected_arm].step()  # type: ignore
        return next_machine_state, reward

    def get_cur_states(self):
        return np.array([machine.current_state for machine in self.machines])
        
    def reset(self, random=False):
        if not random:
            for machine in self.machines:
                machine.reset()
        else:
            for machine in self.machines:
                machine.current_state = int(np.random.choice(self.n))


if __name__ == "__main__":
    mab = MAB(num_arms = test2.num_arms,  
              num_states_per_arm = test2.num_states_per_arm,
              transition_matrices = test2.transition_matrices,
              reward_matrices = test2.reward_matrices,
              initial_start_state=test2.initial_start_state,
              homogeneous=test2.homogeneous)
    print(mab.step(1))
    print(mab.homogeneous)
    mab.reset(random=True)
    print(mab.get_cur_states())
