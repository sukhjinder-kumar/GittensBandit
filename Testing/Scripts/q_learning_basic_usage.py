from Environment.mab_environment import Mab
from Environment.test_cases import test1
from Strategies.q_learning import QLearning
from Utils.cal_optimal_rewrad import calculate_optimal_reward
from Utils.regret_calculation import calculate_regret, plot_regret_history_average, plot_cumm_regret_average
import numpy as np
from tqdm import tqdm

num_epochs = 10
episode_len = 20
num_runs = 2
test = test1
test_string = "test1"
init_learning_rate = 0.5

mab = Mab(num_arms = test.num_arms,
          num_states_per_arm = test.num_states_per_arm,
          transition_matrices = test.transition_matrices,
          reward_matrices = test.reward_matrices,
          initial_start_state = test.initial_start_state,
          homogeneous = test.homogeneous)

qlearning = QLearning(num_arms = test.num_arms, 
                      num_states_per_arm = test.num_states_per_arm,
                      init_learning_rate = init_learning_rate,
                      discount_factor = test.discount_factor, 
                      temperature_mode = "epsilon-greedy-gittin")

# gittin_history = np.zeros((NUM_RUNS, TIME_STEPS, test2.num_arms, test2.num_states_per_arm))

gittin_history = np.zeros((num_runs, num_epochs, test.num_arms, test.num_states_per_arm))
optimal_reward = calculate_optimal_reward(test, mab, episode_len)
regret_history = np.zeros((num_runs, num_epochs))
cumm_regret = np.zeros((num_epochs))

for run in tqdm(range(num_runs), unit=" #Run"):
    qlearning.reset()
 
    for epoch in range(num_epochs):
        mab.reset(random=True)
        for t in range(episode_len):
            cur_states = mab.get_cur_state()
            cur_action, _ = qlearning.get_action(cur_states)
            next_state, cur_reward = mab.step(cur_action)
            qlearning.update(cur_states[cur_action], next_state, cur_reward, cur_action)

        # Store gittins
        for k in range(test.num_arms):
            for n in range(test.num_states_per_arm):
                gittin_history[run, epoch, k, n] = qlearning.q_table[n, 0, n, k]

        # calculate regret
        regret_history[run][epoch] = calculate_regret(mab, qlearning, episode_len, optimal_reward, test.discount_factor)

print("Training Finished!")
save_path = f'Results/QLearning/{test_string}_gittin_plot_num_runs={num_runs}_time_steps={num_epochs}.png'
qlearning.qlearning_visualize(gittin_history, "gittin plot", save_path)
print("Plot saved")

regret_history_average = np.mean(regret_history, axis=0)  # shape: (num_episodes)
regret_savepath = f"Results/Reinforce/regret_plot_{test_string}_num_epochs={num_epochs}_num_runs={num_runs}_init_lr={init_learning_rate}_episode_len={episode_len}.png"
plot_regret_history_average(regret_history_average,
                            title=f"Average cumm regret over a episode_len={episode_len}",
                            savepath=regret_savepath)

cumm_regret_counter = 0
for epoch in range(num_epochs):
    cumm_regret_counter += regret_history_average[epoch]
    cumm_regret[epoch] = cumm_regret_counter

cumm_regret_savepath = f"Results/Reinforce/cumm_regret_plot_{test_string}_num_epochs={num_epochs}_num_runs={num_runs}_init_lr={init_learning_rate}_episode_len={episode_len}.png"
plot_cumm_regret_average(cumm_regret,
                         title=f"Average cummulative regret over a episode_len={episode_len}",
                         savepath=regret_savepath)
