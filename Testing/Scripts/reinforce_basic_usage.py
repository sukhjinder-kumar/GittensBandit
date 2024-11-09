import numpy as np
from Environment.test_cases import test1, test2, test3
from Environment.mab_environment import Mab
from Strategies.reinforce import Reinforce
from Utils.cal_optimal_rewrad import calculate_optimal_reward
from Utils.regret_calculation import calculate_regret, plot_regret_history_average, plot_cumm_regret_average
from tqdm import tqdm

num_epochs = 2
episode_len = 20
num_runs = 2
learning_rate = 0.001
temperature = 200
test = test1
test_string = "test1"
schedule = "linear"

mab = Mab(num_arms = test.num_arms,
          num_states_per_arm = test.num_states_per_arm,
          transition_matrices = test.transition_matrices,
          reward_matrices = test.reward_matrices,
          initial_start_state = test.initial_start_state,
          homogeneous = test.homogeneous)

reinforce = Reinforce(num_arms = test.num_arms,
                      num_states_per_arm = test.num_states_per_arm,
                      homogeneous=test.homogeneous,
                      discount_factor = test.discount_factor,
                      episode_len = episode_len, 
                      learning_rate = learning_rate,
                      temperature = temperature,
                      schedule = schedule)

optimal_reward = calculate_optimal_reward(test, mab, episode_len)
regret_history = np.zeros((num_runs, num_epochs))
cumm_regret = np.zeros((num_epochs))

# Code for homogeneous
if mab.homogeneous:
    h_history = np.zeros((num_runs, num_epochs, mab.n))

    for run in tqdm(range(num_runs), unit="#runs"):
        reinforce.reset()
        for epoch in range(num_epochs):
            mab.reset(random=True)
            state_history = np.zeros((episode_len, mab.k), dtype=int)
            action_history = np.zeros((episode_len), dtype=int)
            action_probability_history = np.zeros((episode_len, mab.k))
            reward_history = np.zeros((episode_len))
            
            # generate episode
            for t in range(episode_len):
                cur_state = mab.get_cur_state()
                action, action_probability = reinforce.get_action(cur_state)
                _, reward = mab.step(action)
                
                state_history[t] = cur_state.copy()
                action_history[t] = action
                action_probability_history[t] = action_probability.copy()
                reward_history[t] = reward

            # Compute cumulative reward G_t for each time step
            G = 0
            return_history = np.zeros((episode_len))  # = \sum_t^T R_t
            for t, reward in enumerate(reversed(reward_history)):
                G = reward + test.discount_factor * G
                return_history[episode_len - t - 1] = G
        
            # update reinforce
            for t in range(episode_len):
                reinforce.update(cur_state=state_history[t],
                                 next_state=None,
                                 reward=reward_history[t], 
                                 action_taken=action_history[t],
                                 action_probability=action_probability_history[t],
                                 cumm_reward=return_history[t],
                                 cur_time=t)

            # update h_history
            h_history[run, epoch] = reinforce.h.copy()

            # calculate regret
            regret_history[run][epoch] = calculate_regret(mab, reinforce, episode_len, optimal_reward, test.discount_factor)

# Code for non-homogeneous
else:
    h_history = np.zeros((num_runs, num_epochs, mab.k, mab.n))

    for run in tqdm(range(num_runs), unit="#runs"):
        reinforce.reset()
        for epoch in range(num_epochs):
            mab.reset(random=True)
            state_history = np.zeros((episode_len, mab.k), dtype=int)
            action_history = np.zeros((episode_len), dtype=int)
            action_probability_history = np.zeros((episode_len, mab.k))
            reward_history = np.zeros((episode_len))
            
            # generate episode
            for t in range(episode_len):
                cur_state = mab.get_cur_state()
                action, action_probability = reinforce.get_action(cur_state)
                _, reward = mab.step(action)
                
                state_history[t] = cur_state.copy()
                action_history[t] = action
                action_probability_history[t] = action_probability.copy()
                reward_history[t] = reward

            # Compute cumulative reward G_t for each time step
            G = 0
            return_history = np.zeros((episode_len))  # = \sum_t^T R_t
            for t, reward in enumerate(reversed(reward_history)):
                G = reward + test.discount_factor * G
                return_history[episode_len - t - 1] = G
        
            # update reinforce
            for t in range(episode_len):
                reinforce.update(cur_state=state_history[t],
                                 next_state=None,
                                 action_taken=action_history[t],
                                 action_probability=action_probability_history[t],
                                 reward=reward_history[t], 
                                 cumm_reward=return_history[t],
                                 cur_time=t)

            # update h_history
            h_history[run, epoch] = reinforce.h.copy()

            # calculate regret
            regret_history[run][epoch] = calculate_regret(mab, reinforce, episode_len, optimal_reward, test.discount_factor)

h_average = np.mean(h_history, axis=0)  # shape: (num_episode, k, n)
title = f"Average Preference vs episode, num_epochs: {num_epochs} and num_runs: {num_runs}"
preference_savepath = f"Results/Reinforce/preference_plot_{test_string}_num_epochs={num_epochs}_num_runs={num_runs}_lr={learning_rate}_episode_len={episode_len}_temp={temperature}.png"
reinforce.visualize_h_average(h_average,
                              title=title,
                              savepath=preference_savepath)

regret_history_average = np.mean(regret_history, axis=0)  # shape: (num_episodes)
regret_savepath = f"Results/Reinforce/regret_plot_{test_string}_num_epochs={num_epochs}_num_runs={num_runs}_lr={learning_rate}_episode_len={episode_len}_temp={temperature}.png"
plot_regret_history_average(regret_history_average,
                            title=f"Average cumm regret over a episode_len={episode_len}",
                            savepath=regret_savepath)

cumm_regret_counter = 0
for epoch in range(num_epochs):
    cumm_regret_counter += regret_history_average[epoch]
    cumm_regret[epoch] = cumm_regret_counter

cumm_regret_savepath = f"Results/Reinforce/cumm_regret_plot_{test_string}_num_epochs={num_epochs}_num_runs={num_runs}_lr={learning_rate}_episode_len={episode_len}_temp={temperature}.png"
plot_cumm_regret_average(cumm_regret,
                         title=f"Average cummulative regret over a episode_len={episode_len}",
                         savepath=cumm_regret_savepath)
