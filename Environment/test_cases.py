import numpy as np


class TestCases():
    def __init__(self, num_arms, 
                 num_states_per_arm, 
                 discount_factor, 
                 transition_matrices, 
                 reward_matrices, 
                 initial_start_state,
                 homogeneous,
                 ranking):
        self.num_arms = num_arms
        self.num_states_per_arm = num_states_per_arm
        self.discount_factor = discount_factor
        self.transition_matrices = transition_matrices
        self.reward_matrices = reward_matrices
        self.initial_start_state = initial_start_state
        self.homogeneous = homogeneous
        self.ranking = ranking  # ordering of (arm, state) or just (state), for homogeneous, in descending order of rank

    def get_highest_rank(self, cur_state):
        if cur_state.shape[0] != self.num_arms:
            raise Exception("Entered wrong cur_state, > test.num_arms")

        # return arm with best rank
        ranks = np.zeros((self.num_arms), dtype=int)
        for arm, cur_arm_state in enumerate(cur_state):
            if self.homogeneous:
                ranks[arm] = self.ranking[(cur_arm_state)]
            else:
                ranks[arm] = self.ranking[(arm, cur_arm_state)]
        return int(np.argmin(ranks))


def compute_gittins_index(transition_matrix, reward_vector, discount_factor):
    """
    Computes the Gittins index for a Markov chain.
    Largest Remaining Index Algorithm by Varaiya, Walrand and Buyukkoc

    Parameters:
        transition_matrix (numpy.ndarray): A square matrix of shape (num_states, num_states)
            where entry (i, j) represents the probability of transitioning from state i to state j.
        reward_matrix (numpy.ndarray): A nparray of shape (num_states)
        discount_factor (float): The discount factor (0 < discount_factor < 1).

    Returns:
        numpy.ndarray: A vector of Gittins indices for each state.
    """

    num_states = transition_matrix.shape[0]
    gittin_index = np.zeros((num_states), dtype=float)
    Q = np.zeros((num_states, num_states), dtype=float)
    S_set = set(range(num_states))
    C_set = set()

    # 1: Find maximum reward
    best_gittin = (float)(np.max(reward_vector))
    best_state = (int)(np.argmax(reward_vector))
    gittin_index[best_state] = best_gittin
    C_set.add(best_state)
    S_set.remove(best_state)

    # 2. Recursively find other gittins
    for i in range(1, num_states, 1):

        # Update Q
        for col_num in C_set:
            Q[:, col_num] = transition_matrix[:, col_num]

        # Generate d vector
        t = np.eye(num_states) - discount_factor*Q
        d = np.linalg.inv(t) @ reward_vector

        # Generate b vector
        b = np.linalg.inv(t) @ np.ones((num_states)) 

        # Find new best gittin
        l = [d[i]/b[i] for i in S_set]
        new_best_gittin = max(l)
        new_best_state = 0
        for i in S_set:
            if d[i]/b[i] == new_best_gittin:
                new_best_state = i
                break

        # store
        gittin_index[new_best_state] = new_best_gittin
        C_set.add(new_best_state)
        S_set.remove(new_best_state)

    return gittin_index

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
ranking_test_1 = {
    (0,0) : 2,
    (0,1) : 3,
    (1,0) : 4,
    (1,1) : 1
}
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
                  homogeneous=False,
                  ranking=ranking_test_1)

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
ranking_test_2 = {
    (0) : 1,
    (1) : 2,
    (2) : 3
}
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
                  homogeneous=True,
                  ranking=ranking_test_2)

###################################
## 3.
###################################

# Preference -
# 0, 3, 5, 2, 1, 6, 4, 7, 8, 9

num_arms_test_3 = 2
num_states_per_arm_test_3 = 10
discount_factor_test_3 = 0.9
initial_start_state_test_3 = np.array([0, 0], dtype=int)
ranking_test_3 = {
    (0) : 1,
    (3) : 2,
    (5) : 3,
    (2) : 4,
    (1) : 5,
    (6) : 6,
    (4) : 7,
    (7) : 8,
    (8) : 9,
    (9) : 10 
}
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
                  homogeneous=True,
                  ranking=ranking_test_3)

###################################
## 4.
###################################

num_arms_test_4 = 100
num_states_per_arm_test_4 = 100
discount_factor_test_4 = 0.75
initial_start_state_test_4 = np.zeros((num_arms_test_4), dtype=int)

# Generate a random transition matrix for each arm-state pair
tms_test_4 = np.zeros((num_arms_test_4, num_states_per_arm_test_4, num_states_per_arm_test_4))
for arm in range(num_arms_test_4):
    for state in range(num_states_per_arm_test_4):
        row = np.random.rand(num_states_per_arm_test_4)
        tms_test_4[arm, state] = row / row.sum()  # Normalize to ensure probabilities sum to 1

# Generate random reward matrices for each arm-state pair. We here have all row element same for
# offline algorithms to work
rms_test_4 = np.zeros((num_arms_test_4, num_states_per_arm_test_4, num_states_per_arm_test_4))
for i in range(num_arms_test_4):
    for row in range(num_states_per_arm_test_4):
        # generate random number
        temp = np.random.choice(30)
        for col in range(num_states_per_arm_test_4):
            rms_test_4[i][row][col] = temp

# Compute rankings based on Gittins indices
ranking_test_4 = {}
gittins_indices = {}

for arm in range(num_arms_test_4):
    gittins_indices[arm] = compute_gittins_index(
        tms_test_4[arm], 
        rms_test_4[arm][:,0], 
        discount_factor_test_4
    )

all_indices = []
for arm in range(num_arms_test_4):
    for state in range(num_states_per_arm_test_4):
        all_indices.append(((arm, state), gittins_indices[arm][state]))

# Sort by Gittins index in descending order
all_indices.sort(key=lambda x: x[1], reverse=True)

# Assign ranks
for rank, ((arm, state), _) in enumerate(all_indices, start=1):
    ranking_test_4[(arm, state)] = rank

test4 = TestCases(num_arms=num_arms_test_4,
                  num_states_per_arm=num_states_per_arm_test_4,
                  discount_factor = discount_factor_test_4,
                  transition_matrices=tms_test_4,
                  reward_matrices=rms_test_4,
                  initial_start_state=initial_start_state_test_4,
                  homogeneous=False,
                  ranking=ranking_test_4)


###################################
## 5.
###################################

# Preference -
# 2, 0, 1, 3

num_arms_test_5 = 2
num_states_per_arm_test_5 = 4
discount_factor_test_5 = 0.75
initial_start_state_test_5 = np.zeros((num_arms_test_5), dtype=int)

tms_test_5 = np.array([[0.1, 0, 0.8, 0.1],
                       [0.5, 0, 0.1, 0.4],
                       [0.2, 0.6, 0, 0.2],
                       [0, 0.8, 0, 0.2]])

rms_test_5 = np.array([[16, 16, 16, 16],
                       [19, 19, 19, 19],
                       [30, 30, 30, 30],
                       [4, 4, 4, 4]])

# Compute rankings based on Gittins indices
ranking_test_5 = {}
gittins_indices = {}

gittins_indices = compute_gittins_index(
    tms_test_5, 
    rms_test_5[:,0], 
    discount_factor_test_5
)

all_indices = []
for state in range(num_states_per_arm_test_5):
    all_indices.append(((state), gittins_indices[state]))

# Sort by Gittins index in descending order
all_indices.sort(key=lambda x: x[1], reverse=True)

# Assign ranks
for rank, ((state), _) in enumerate(all_indices, start=1):
    ranking_test_5[(state)] = rank

test5 = TestCases(num_arms=num_arms_test_5,
                  num_states_per_arm=num_states_per_arm_test_5,
                  discount_factor = discount_factor_test_5,
                  transition_matrices=tms_test_5,
                  reward_matrices=rms_test_5,
                  initial_start_state=initial_start_state_test_5,
                  homogeneous=True,
                  ranking=ranking_test_5)

if __name__ == "__main__":
    # print(test3.initial_start_state) 
    # cur_state = np.array([0, 1])
    # print(test1.ranking)
    # print(test1.get_highest_rank(cur_state))
    print(ranking_test_5)
    # print(test5.get_highest_rank(np.array((2, 0), dtype=int)))
    # print(rms_test_4[1])
    # print(ranking_test_4)