import numpy as np


class MP:  # markovian process
    def __init__(self,
                 num_states,
                 transition_matrix,
                 reward_matrix,
                 start_state):
        '''
        transition_matrix : np.array((num_states, num_states)) : tm[i][j] = Prob of going i -> j 
            Constraint: sum_j (tm[i][j]) = 1
        reward_matrix : np.array((num_states, num_states))
        '''
        if not self.check_tm(transition_matrix):
            raise Exception("transition matrix doens't sum to 1")
        self.num_states = num_states
        self.transition_matrix = transition_matrix
        # if start_state is not None:
        self.start_state = start_state
        # else:
        #     self.start_state = int(np.random.choice(self.num_states))  # Random initial state
        self.current_state = self.start_state
        self.reward_matrix = reward_matrix

    def step(self):
        transition_probs = self.transition_matrix[self.current_state]
        next_state = int(np.random.choice(self.num_states, p=transition_probs))
        reward = int(self.reward_matrix[self.current_state][next_state])
        self.current_state = next_state
        return next_state, reward

    def check_tm(self, tm):
        row_sums = np.isclose(np.sum(tm, axis=1), 1)
        return np.all(row_sums)  # checks if all element of array are True

    def reset(self):
        self.current_state = self.start_state
