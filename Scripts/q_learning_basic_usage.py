from Environment.mab_environment import Mab
from Environment.test_cases import test1
from Strategies.q_learning import QLearning
import numpy as np
from tqdm import tqdm

num_epochs = 10000
num_runs = 2
test = test1
test_string = "test1"

mab = Mab(num_arms = test.num_arms,
          num_states_per_arm = test.num_states_per_arm,
          transition_matrices = test.transition_matrices,
          reward_matrices = test.reward_matrices,
          initial_start_state = test.initial_start_state,
          homogeneous = test.homogeneous)

qlearning = QLearning(num_arms = test.num_arms, 
                                            num_states_per_arm = test.num_states_per_arm,
                                            init_learning_rate = 0.5,
                                            discount_factor = test.discount_factor, 
                                            temperature_mode = "epsilon-greedy-gittin")

# gittin_history = np.zeros((NUM_RUNS, TIME_STEPS, test2.num_arms, test2.num_states_per_arm))

gittin_history = np.zeros((num_runs, num_epochs, test.num_arms, test.num_states_per_arm))

for num_run in tqdm(range(num_runs), unit=" #Run"):
        mab.reset(random=True)
        qlearning.reset()
        
        for epoch in range(num_epochs):
                cur_states = mab.get_cur_states()
                cur_action, _ = qlearning.get_action(cur_states)
                next_state, cur_reward = mab.step(cur_action)
                qlearning.update(cur_states[cur_action], cur_action, cur_reward, next_state)

                # Store gittins
                for k in range(test.num_arms):
                        for n in range(test.num_states_per_arm):
                                gittin_history[num_run, epoch, k, n] = qlearning.q_table[n, 0, n, k]

print("Training Finished!")
save_path = f'Results/QLearning/{test_string}_gittin_plot_num_runs={num_runs}_time_steps={num_epochs}.png'
qlearning.qlearning_visualize(gittin_history, save_path)
print("Plot saved")
