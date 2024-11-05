import numpy as np
from Environment.test_cases import test1, test2, test3
from Environment.mab_environment import Mab
from Strategies.reinforce import Reinforce
from tqdm import tqdm

num_epochs = 10000
episode_len = 20
num_runs = 2
learning_rate = 0.001
temperature = 1
test = test3
test_string = "test3"

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
                      temperature = temperature)

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
                                 action_taken=action_history[t],
                                 action_probability=action_probability_history[t],
                                 reward=reward_history[t], 
                                 cumm_reward=return_history[t],
                                 cur_time=t)

            # update h_history
            h_history[run, epoch] = reinforce.h.copy()

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
                                 action_taken=action_history[t],
                                 action_probability=action_probability_history[t],
                                 reward=reward_history[t], 
                                 cumm_reward=return_history[t],
                                 cur_time=t)

            # update h_history
            h_history[run, epoch] = reinforce.h.copy()

h_average = np.mean(h_history, axis=0)  # shape: (num_episode, k, n)
title = f"Average Preference vs episode, num_epochs: {num_epochs} and num_runs: {num_runs}"
savepath = f"Results/Reinforce/{test_string}_num_epochs={num_epochs}_num_runs={num_runs}_lr={learning_rate}_episode_len={episode_len}.png"
reinforce.visualize_h_average(h_average,
                              title=title,
                              savepath=savepath)
