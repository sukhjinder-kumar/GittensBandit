import numpy as np


class TestCases():
    def __init__(self, num_arms, 
                 num_states_per_arm, 
                 discount_factor, 
                 transition_matrices, 
                 reward_matrices, 
                 initial_start_state,
                 homogeneous):
        self.num_arms = num_arms
        self.num_states_per_arm = num_states_per_arm
        self.discount_factor = discount_factor
        self.transition_matrices = transition_matrices
        self.reward_matrices = reward_matrices
        self.initial_start_state = initial_start_state
        self.homogeneous = homogeneous

###################################
## 1. (Based on Duff 1995)
###################################

# G(0,0) = ~24
# G(0,1) = ~16
# G(1,0) = ~10
# G(1,1) = ~30

num_arms_test_1 = 2
num_states_per_arm_test_1 = 2
discount_factor_test_1 = 0.7
initial_start_state_test_1 = np.array([0, 0], dtype=int)
tms_test_1 = np.array([[[0.3, 0.7],  # transition_matrices_test_1
                        [0.7, 0.3]],
                       
                       [[0.9, 0.1],
                        [0.1, 0.9]]])

rms_test_1 = np.array([[[1, 10],  # reward_matrices_test_1
                        [1, 10]],
                       
                       [[1, 10],
                        [ 1, 10]]])

test1 = TestCases(num_arms=num_arms_test_1,
                  num_states_per_arm=num_states_per_arm_test_1,
                  discount_factor = discount_factor_test_1,
                  transition_matrices=tms_test_1,
                  reward_matrices=rms_test_1,
                  initial_start_state=initial_start_state_test_1,
                  homogeneous=False)

###################################
## 2. (based on proj. tejas paper)
###################################

# G(0,0) = 0.9
# G(0,1) = 0.814
# G(0,2) = 0.758
num_arms_test_2 = 2
num_states_per_arm_test_2 = 3
discount_factor_test_2 = 0.9
initial_start_state_test_2 = np.array([0, 0], dtype=int)
tms_test_2 = np.array([[0.1, 0.9, 0],
                       [0.1, 0, 0.9],
                       [0.1, 0, 0.9]])  # Homogeneous arms

rms_test_2 = np.array([[2, 2, 2],  # r(s) = 0.9^s + 1
                       [1.9, 1.9, 1.9],
                       [1.81, 1.81, 1.81]])

test2 = TestCases(num_arms=num_arms_test_2,
                  num_states_per_arm=num_states_per_arm_test_2,
                  discount_factor = discount_factor_test_2,
                  transition_matrices=tms_test_2,
                  reward_matrices=rms_test_2,
                  initial_start_state=initial_start_state_test_2,
                  homogeneous=True)
###################################
## 3.
###################################

# Preference -
# 0, 3, 5, 2, 1, 6, 4, 7, 8, 9
# See image in figures/q_learning for gittin plot

num_arms_test_3 = 2
num_states_per_arm_test_3 = 10
discount_factor_test_3 = 0.9
initial_start_state_test_3 = np.zeros((2))

tms_test_3 = np.array([
    [0.1, 0.2, 0.1, 0.05, 0.1, 0.05, 0.15, 0.05, 0.1, 0.1],
    [0.05, 0.1, 0.2, 0.1, 0.05, 0.1, 0.05, 0.15, 0.1, 0.1],
    [0.1, 0.05, 0.1, 0.15, 0.1, 0.05, 0.1, 0.2, 0.05, 0.1],
    [0.1, 0.1, 0.05, 0.2, 0.1, 0.1, 0.05, 0.05, 0.1, 0.15],
    [0.2, 0.1, 0.1, 0.05, 0.1, 0.1, 0.05, 0.1, 0.1, 0.1],
    [0.15, 0.05, 0.1, 0.1, 0.2, 0.05, 0.1, 0.1, 0.1, 0.05],
    [0.05, 0.1, 0.05, 0.1, 0.1, 0.2, 0.15, 0.05, 0.1, 0.1],
    [0.1, 0.05, 0.15, 0.1, 0.05, 0.1, 0.1, 0.2, 0.05, 0.1],
    [0.1, 0.1, 0.05, 0.05, 0.1, 0.15, 0.1, 0.05, 0.2, 0.1],
    [0.05, 0.15, 0.1, 0.05, 0.1, 0.1, 0.05, 0.1, 0.1, 0.2]
])

rms_test_3 = np.array([
    [0, 1, -1, 0.5, 0, -0.5, 2, -1, 0, 1],
    [0.5, 0, 1, -0.5, 0, 1.5, -2, 0.5, 0, 0.2],
    [-1, 0, 2, 0.3, 1, 0.5, 0, -0.5, 0.8, 0],
    [0.1, 0.5, -0.8, 0, 1.2, 0.4, -0.2, 0.9, 0, 1.1],
    [0.4, -0.5, 0.7, 0.3, 0, -0.1, 1, 0, -0.6, 0.2],
    [1, -0.2, 0, 0.6, 0.3, 0, 1.4, -0.9, 0.5, -1],
    [-0.3, 1, 0.8, 0, 0.5, -0.6, 0, 1.3, -0.2, 0.7],
    [0.2, -1, 0.9, 0.7, 0, 1, -0.3, 0, 0.6, -0.4],
    [0.5, 0, -0.7, 0.3, 0.1, 0.9, -0.5, 1, 0, 0.2],
    [-0.8, 0.3, 0.1, -0.9, 0.6, 0, 1, -0.2, 0.4, 0]
])

test3 = TestCases(num_arms=num_arms_test_3,
                  num_states_per_arm=num_states_per_arm_test_3,
                  discount_factor = discount_factor_test_3,
                  transition_matrices=tms_test_3,
                  reward_matrices=rms_test_3,
                  initial_start_state=initial_start_state_test_3,
                  homogeneous=True)


if __name__ == "__main__":
    for row_idx in range(10):
        print(sum(tms_test_3[row_idx,:]))