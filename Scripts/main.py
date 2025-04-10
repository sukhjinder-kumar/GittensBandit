import numpy as np
from tqdm import tqdm

from Environment.test_cases import test1, test2, test3, test4, test5
from Environment.mab_environment import Mab
from Strategies.reinforce_with_adam import ReinforceWithAdam
from Strategies.reinforce_with_momentum import ReinforceWithMomentum
from Utils.cal_optimal_reward import calculate_optimal_reward, plot_regret_history_average, plot_cumm_regret_average, calculate_cumm_reward
from Scripts.argparser_config import get_args
from Strategies.q_learning import QLearning
from Strategies.reinforce import Reinforce

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

# QLearning
if mab.homogeneous:
    num_arms = 1
else:
    num_arms = test.num_arms

# if args.qlearning_schedule == "Boltzmann":
#     strategy = QLearning(num_arms = num_arms, 
#                         num_states_per_arm = test.num_states_per_arm,
#                         discount_factor = test.discount_factor,
#                         init_learning_rate = args.qlearning_init_learning_rate,
#                         tau = args.qlearning_tau,
#                         schedule = args.qlearning_schedule,
#                         max_temperature = args.qlearning_max_temperature,
#                         min_temperature = args.qlearning_min_temperature,
#                         beta = args.qlearning_beta)
# elif args.qlearning_schedule == "epsilon-greedy":
#     strategy = QLearning(num_arms = num_arms, 
#                         num_states_per_arm = test.num_states_per_arm,
#                         discount_factor = test.discount_factor,
#                         init_learning_rate = args.qlearning_init_learning_rate,
#                         tau = args.qlearning_tau,
#                         schedule = args.qlearning_schedule,
#                         epsilon_greedy = args.qlearning_epsilon_greedy)

# strategies.append(strategy)
gittin_history = np.zeros((args.num_runs, args.num_epochs, num_arms, test.num_states_per_arm))

# Reinforce with linear and none schedule
strategy = Reinforce(num_arms=test.num_arms,
                    num_states_per_arm=test.num_states_per_arm,
                    homogeneous=mab.homogeneous,
                    discount_factor=test.discount_factor,
                    learning_rate=args.reinforce_learning_rate,
                    schedule="linear",
                    max_temperature=args.reinforce_max_temperature,
                    min_temperature=args.reinforce_min_temperature,
                    beta=args.reinforce_beta,
                    name="Reinforce_Linear")
strategies.append(strategy)

strategy = Reinforce(num_arms=test.num_arms,
                    num_states_per_arm=test.num_states_per_arm,
                    homogeneous=mab.homogeneous,
                    discount_factor=test.discount_factor,
                    learning_rate=args.reinforce_learning_rate,
                    schedule="none",
                    constant_temperature=args.reinforce_constant_temperature,
                    name="Reinforce_None")
strategies.append(strategy)

# Reinforce with momentum with linear and none schedule
strategy = ReinforceWithMomentum(num_arms=test.num_arms,
                    num_states_per_arm=test.num_states_per_arm,
                    homogeneous=mab.homogeneous,
                    discount_factor=test.discount_factor,
                    learning_rate=args.reinforce_learning_rate,
                    schedule="linear",
                    max_temperature=args.reinforce_max_temperature,
                    min_temperature=args.reinforce_min_temperature,
                    beta=args.reinforce_beta,
                    name="Reinforce_Linear")
strategies.append(strategy)

strategy = ReinforceWithMomentum(num_arms=test.num_arms,
                    num_states_per_arm=test.num_states_per_arm,
                    homogeneous=mab.homogeneous,
                    discount_factor=test.discount_factor,
                    learning_rate=args.reinforce_learning_rate,
                    schedule="none",
                    constant_temperature=args.reinforce_constant_temperature,
                    name="Reinforce_None")
strategies.append(strategy)

# Reinforce with adam with linear and none schedule
strategy = ReinforceWithAdam(num_arms=test.num_arms,
                    num_states_per_arm=test.num_states_per_arm,
                    homogeneous=mab.homogeneous,
                    discount_factor=test.discount_factor,
                    learning_rate=args.reinforce_learning_rate,
                    schedule="linear",
                    max_temperature=args.reinforce_max_temperature,
                    min_temperature=args.reinforce_min_temperature,
                    beta=args.reinforce_beta,
                    name="Reinforce_Linear")
strategies.append(strategy)

strategy = ReinforceWithAdam(num_arms=test.num_arms,
                    num_states_per_arm=test.num_states_per_arm,
                    homogeneous=mab.homogeneous,
                    discount_factor=test.discount_factor,
                    learning_rate=args.reinforce_learning_rate,
                    schedule="none",
                    constant_temperature=args.reinforce_constant_temperature,
                    name="Reinforce_None")
strategies.append(strategy)

if mab.homogeneous:
    h_history_linear = np.zeros((args.num_runs, args.num_epochs, mab.n))
    h_history_none = np.zeros((args.num_runs, args.num_epochs, mab.n))
else:
    h_history_linear = np.zeros((args.num_runs, args.num_epochs, mab.k, mab.n))
    h_history_none = np.zeros((args.num_runs, args.num_epochs, mab.k, mab.n))

# Other Strategies
# ...

# init common variables
optimal_reward = calculate_optimal_reward(test, mab, args.episode_len)
regret_history = np.zeros((len(strategies), args.num_runs, args.num_epochs))
avg_cumm_regret = np.zeros((len(strategies), args.num_epochs))

# Main Logic
for idx_strategy, strategy in enumerate(strategies):
    for run in tqdm(range(args.num_runs), unit=f" #{strategy.name}-Run"):
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

            strategy.long_term_update(state_history,
                                      next_state_history,
                                      reward_history,
                                      action_taken_history,
                                      action_probability_history,
                                      args.episode_len)

            if strategy.name == "QLearning" and not args.qlearning_not_show_gittin_plot:
                for k in range(num_arms):
                    for n in range(test.num_states_per_arm):
                        gittin_history[run, epoch, k, n] = strategy.q_table[n, 0, n, k]
            
            if strategy.name == "Reinforce_Linear" and not args.reinforce_not_show_preference_plot:
                h_history_linear[run, epoch] = strategy.h.copy()

            if strategy.name == "Reinforce_None" and not args.reinforce_not_show_preference_plot:
                h_history_none[run, epoch] = strategy.h.copy()

            # calculate regret
            if not args.not_show_regret_average_plot:
                regret_history[idx_strategy][run][epoch] = calculate_cumm_reward(strategy, test, mab, args.episode_len) - optimal_reward

for strategy in strategies:
    if strategy.name == "QLearning":
        if not args.qlearning_not_show_gittin_plot:
            title = f'Gittins index for num_runs={args.num_runs} and num_epochs={args.num_epochs}'
            savepath = f'{args.savepath}/gittin_plot.png'
            strategy.qlearning_visualize(gittin_history, title, savepath)

    if strategy.name == "Reinforce_Linear":
        if not args.reinforce_not_show_preference_plot:
            h_average = np.mean(h_history_linear, axis=0)
            title = f"Reinforce (Linear): Preference vs episode, num_epochs: {args.num_epochs} and num_runs: {args.num_runs}"
            savepath = f"{args.savepath}/linear_preference_plot.png"
            strategy.visualize_h_average(h_average=h_average,
                                        title=title,
                                        savepath=savepath)

    if strategy.name == "Reinforce_None":
        if not args.reinforce_not_show_preference_plot:
            h_average = np.mean(h_history_none, axis=0)
            title = f"Reinforce (None): Average Preference vs episode, num_epochs: {args.num_epochs} and num_runs: {args.num_runs}"
            savepath = f"{args.savepath}/none_preference_plot.png"
            strategy.visualize_h_average(h_average=h_average,
                                        title=title,
                                        savepath=savepath)

# Plot regret history
regret_history_average = np.mean(regret_history, axis=1)  # shape: (len(strategies), num_episodes)
if not args.not_show_regret_average_plot:
    regret_savepath = f'{args.savepath}/regret_plot.png'
    plot_regret_history_average(regret_history_average,
                                strategies,
                                title=f"Average regret with num_runs={args.num_runs} and episode_len={args.episode_len}",
                                savepath=regret_savepath)

# Plot cumm regret
if not args.not_show_cumm_regret_average_plot:
    for i in range(len(strategies)):
        cumm_regret_counter = 0
        for epoch in range(args.num_epochs):
            cumm_regret_counter += regret_history_average[i][epoch]
            avg_cumm_regret[i][epoch] = cumm_regret_counter

    cumm_regret_savepath = f'{args.savepath}/cumm_regret_plot.png'
    plot_cumm_regret_average(avg_cumm_regret,
                             strategies,
                             title=f"Average cumm regret with num_runs={args.num_runs} and episode_len={args.episode_len}",
                             savepath=cumm_regret_savepath)