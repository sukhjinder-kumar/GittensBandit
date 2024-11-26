import numpy as np
from tqdm import tqdm
from Strategies.neural_reinforce import NeuralReinforce
from Utils.regret_calculation import calculate_regret

def neural_reinforce_train(args,
                           mab,
                           test,
                           optimal_reward,
                           regret_history):

    nn_reinforce = NeuralReinforce(num_arms = test.num_arms,
                                   num_states_per_arm = test.num_states_per_arm,
                                   homogeneous = test.homogeneous,
                                   discount_factor = test.discount_factor,
                                   episode_len = args.episode_len, 
                                   learning_rate = args.learning_rate,
                                   max_temperature = args.temperature,
                                   schedule="linear")

    if mab.homogeneous:  # Code for homogeneous
        h_history = np.zeros((args.num_runs, args.num_epochs, mab.n))

        for run in tqdm(range(args.num_runs), unit="#runs"):
            nn_reinforce.reset()
            for epoch in range(args.num_epochs):
                mab.reset(random=True)
                state_history = np.zeros((args.episode_len, mab.k), dtype=int)
                action_history = np.zeros((args.episode_len), dtype=int)
                action_probability_history = np.zeros((args.episode_len, mab.k))
                reward_history = np.zeros((args.episode_len))
                
                # generate episode
                for t in range(args.episode_len):
                        cur_state = mab.get_cur_state()
                        action, action_probability = nn_reinforce.get_action(cur_state)
                        _, reward = mab.step(action)
                        
                        state_history[t] = cur_state.copy()
                        action_history[t] = action
                        action_probability_history[t] = action_probability.copy()
                        reward_history[t] = reward

                # Compute cumulative reward G_t for each time step
                G = 0
                return_history = np.zeros((args.episode_len))  # = \sum_t^T R_t
                for t, reward in enumerate(reversed(reward_history)):
                        G = reward + test.discount_factor * G
                        return_history[args.episode_len - t - 1] = G
                
                # update reinforce
                for t in range(args.episode_len):
                        nn_reinforce.update(cur_state=state_history[t],
                                            next_state=None,
                                            action_taken=action_history[t],
                                            action_probability=action_probability_history[t],
                                            reward=reward_history[t], 
                                            cumm_reward=return_history[t],
                                            cur_time=t)

                # update h_history
                h = np.zeros((mab.n))
                for i in range(mab.n):
                    h[i] = nn_reinforce.h(i).item()
                h_history[run, epoch] = h.copy()

                # calculate regret
                regret_history[run][epoch] = calculate_regret(mab, nn_reinforce, args.episode_len, optimal_reward, test.discount_factor)

    else:  # Code for non-homogeneous
        h_history = np.zeros((args.num_runs, args.num_epochs, mab.k, mab.n))

        for run in tqdm(range(args.num_runs), unit="#runs"):
            nn_reinforce.reset()
            for epoch in range(args.num_epochs):
                mab.reset(random=True)
                state_history = np.zeros((args.episode_len, mab.k), dtype=int)
                action_history = np.zeros((args.episode_len), dtype=int)
                action_probability_history = np.zeros((args.episode_len, mab.k))
                reward_history = np.zeros((args.episode_len))
                
                # generate episode
                for t in range(args.episode_len):
                        cur_state = mab.get_cur_state()
                        action, action_probability = nn_reinforce.get_action(cur_state)
                        _, reward = mab.step(action)
                        
                        state_history[t] = cur_state.copy()
                        action_history[t] = action
                        action_probability_history[t] = action_probability.copy()
                        reward_history[t] = reward

                # Compute cumulative reward G_t for each time step
                G = 0
                return_history = np.zeros((args.episode_len))  # = \sum_t^T R_t
                for t, reward in enumerate(reversed(reward_history)):
                        G = reward + test.discount_factor * G
                        return_history[args.episode_len - t - 1] = G
                
                # update reinforce
                for t in range(args.episode_len):
                        nn_reinforce.update(cur_state=state_history[t],
                                            next_state=None,
                                            action_taken=action_history[t],
                                            action_probability=action_probability_history[t],
                                            reward=reward_history[t], 
                                            cumm_reward=return_history[t],
                                            cur_time=t)

                # update h_history
                h = np.zeros((mab.k, mab.n))
                for i in range(mab.k):
                    for j in range(mab.n):
                        h[i][j] = nn_reinforce.h(i, j).item()
                h_history[run, epoch] = h.copy()

                # calculate regret
                regret_history[run][epoch] = calculate_regret(mab, nn_reinforce, args.episode_len, optimal_reward, test.discount_factor)

    if not args.not_show_preference_plot:
        h_average = np.mean(h_history, axis=0)
        title = f"Average Preference vs episode, num_epochs: {args.num_epochs} and num_runs: {args.num_runs}"
        savepath = f"{args.savepath}/preference_plot.png"
        nn_reinforce.visualize_h_average(h_average=h_average,
                                         title=title,
                                         savepath=savepath)

    return
