import numpy as np

from Environment.test_cases import test1, test2, test3
from Environment.mab_environment import Mab
from Utils.cal_optimal_rewrad import calculate_optimal_reward
from Utils.regret_calculation import plot_regret_history_average, plot_cumm_regret_average
from Scripts.argparser_config import get_args
from Scripts.reinforce_train import reinforce_train
from Scripts.qlearning_train import qlearning_train
from Scripts.neural_reinforce_train import neural_reinforce_train

# get argument from command line
args = get_args()
print(vars(args))

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

# init common variables
optimal_reward = calculate_optimal_reward(test, mab, args.episode_len)
regret_history = np.zeros((args.num_runs, args.num_epochs))
cumm_regret = np.zeros((args.num_epochs))

# Select Strategy
if args.strategy_name == "Reinforce":
    reinforce_train(args,
                    mab,
                    test,
                    optimal_reward,
                    regret_history)  # np.array are passed by reference, so regret_history need not be returned

elif args.strategy_name == "QLearning":
    qlearning_train(args,
                    mab,
                    test,
                    optimal_reward,
                    regret_history)

elif args.strategy_name == "NeuralReinforce":
     neural_reinforce_train(args,
                            mab,
                            test,
                            optimal_reward,
                            regret_history)
else:
    pass

# Plot regret history
regret_history_average = np.mean(regret_history, axis=0)  # shape: (num_episodes)
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
        cumm_regret[epoch] = cumm_regret_counter

    cumm_regret_savepath = f'{args.savepath}/cumm_regret_plot.png'
    plot_cumm_regret_average(cumm_regret,
                             title=f"Average cumm regret with num_runs={args.num_runs} and episode_len={args.episode_len}",
                             savepath=cumm_regret_savepath)
