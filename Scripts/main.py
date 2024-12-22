import numpy as np
from tqdm import tqdm

from Environment.test_cases import test1, test2, test3
from Environment.mab_environment import Mab
from Utils.cal_optimal_rewrad import calculate_optimal_reward
from Utils.regret_calculation import plot_regret_history_average, plot_cumm_regret_average, calculate_regret
from Scripts.argparser_config import get_args
from Scripts.reinforce_train import reinforce_train
from Scripts.qlearning_train import qlearning_train
from Scripts.neural_reinforce_train import neural_reinforce_train
from Strategies.q_learning import QLearning

# get argument from command line
args = get_args()

# Init test case
test = globals()[args.test_name]
test_string = args.test_name

# Init multi arm bandit object (Environment)
mab = Mab(num_arms = test.num_arms,
          num_states_per_arm = test.num_states_per_arm,
          transition_matrices = test.transition_matrices,
          reward_matrices = test.reward_matrices,
          initial_start_state = test.initial_start_state,
          homogeneous = test.homogeneous)

# Select Strategies
strategies = []

if args.strategy_name == "QLearning":
    if mab.homogeneous:
        num_arms = 1
    else:
        num_arms = test.num_arms

    if args.schedule == "Boltzmann":
        strategy = QLearning(num_arms = num_arms, 
                            num_states_per_arm = test.num_states_per_arm,
                            discount_factor = test.discount_factor,
                            init_learning_rate = args.init_learning_rate,
                            tau = args.tau,
                            schedule = args.schedule,
                            max_temperature = args.temperature,
                            min_temperature = args.temperature_mode,
                            beta = args.beta)
    elif args.schedule == "epsilon-greedy":
        strategy = QLearning(num_arms = num_arms, 
                            num_states_per_arm = test.num_states_per_arm,
                            discount_factor = test.discount_factor,
                            init_learning_rate = args.init_learning_rate,
                            tau = args.tau,
                            schedule = args.schedule,
                            epsilon_greedy = args.epsilon_greedy)
    strategies.append(strategy)
    gittin_history = np.zeros((args.num_runs, args.num_epochs, num_arms, test.num_states_per_arm))

elif args.strategy_name == "Reinforce":
    pass
    # reinforce_train(args,
    #                 mab,
    #                 test,
    #                 optimal_reward,
    #                 regret_history)  # np.array are passed by reference, so regret_history need not be returned
elif args.strategy_name == "NeuralReinforce":
    pass
else:
    pass

# init common variables
optimal_reward = calculate_optimal_reward(test, mab, args.episode_len)
regret_history = np.zeros((len(strategies), args.num_runs, args.num_epochs))
avg_cumm_regret = np.zeros((len(strategies), args.num_epochs))

# Main Logic
for idx_strategy, strategy in enumerate(strategies):
    for run in tqdm(range(args.num_runs), unit=" #Run"):
        strategy.reset()
        for epoch in range(args.num_epochs):
            mab.reset(random=True)

            state_history = np.zeros((args.episode_len, test.num_arms), dtype=int)
            next_state_history = np.zeros((args.episode_len), dtype=int)
            reward_history = np.zeros((args.episode_len), dtype=float)
            action_taken_history = np.zeros((args.episode_len), dtype=int)
            action_probability_history = np.zeros((args.episode_len, test.num_arms), dtype=float)
 
            for t in range(args.episode_len):
                cur_states = mab.get_cur_state()
                cur_action, action_probabilities = strategy.get_action(cur_states)
                next_state, reward = mab.step(cur_action)
                strategy.short_term_update(cur_states, 
                                           next_state, 
                                           reward, 
                                           cur_action,
                                           action_probabilities,
                                           t)
                # update state params
                state_history[t] = cur_states 
                next_state_history[t] = next_state
                reward_history[t] = reward
                action_taken_history[t] = cur_action
                action_probability_history[t] = action_probabilities

            # Store gittins
            strategy.long_term_update(state_history,
                                      next_state_history,
                                      reward_history,
                                      action_taken_history,
                                      action_probability_history,
                                      args.episode_len)

            if strategy.name == "QLearning":
                for k in range(num_arms):
                    for n in range(test.num_states_per_arm):
                        gittin_history[run, epoch, k, n] = strategy.q_table[n, 0, n, k]

            # calculate regret
            regret_history[idx_strategy][run][epoch] = calculate_regret(mab, strategy, args.episode_len, optimal_reward, test.discount_factor)
            

if any(strategy.name == "QLearning" for strategy in strategies):
    if not args.not_show_gittin_plot:
        title = f'Gittins index for num_runs={args.num_runs} and num_epochs={args.num_epochs}'
        savepath = f'{args.savepath}/gittin_plot.png'
        strategy.qlearning_visualize(gittin_history, title, savepath)

# Plot regret history
regret_history_average = np.mean(regret_history, axis=1)  # shape: (len(strategies), num_episodes)
if not args.not_show_regret_average_plot:
    regret_savepath = f'{args.savepath}/regret_plot.png'
    plot_regret_history_average(regret_history_average,
                                title=f"Average regret with num_runs={args.num_runs} and episode_len={args.episode_len}",
                                savepath=regret_savepath)

# Plot cumm regret
if not args.not_show_cumm_regret_average_plot:
    cumm_regret_counter = 0
    for epoch in range(args.num_epochs):
        cumm_regret_counter += regret_history_average[epoch]
        avg_cumm_regret[epoch] = cumm_regret_counter

    cumm_regret_savepath = f'{args.savepath}/cumm_regret_plot.png'
    plot_cumm_regret_average(avg_cumm_regret,
                             title=f"Average cumm regret with num_runs={args.num_runs} and episode_len={args.episode_len}",
                             savepath=cumm_regret_savepath)
