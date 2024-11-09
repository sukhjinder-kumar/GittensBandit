import numpy as np
from tqdm import tqdm
from Strategies.q_learning import QLearning
from Utils.regret_calculation import calculate_regret

def qlearning_train(args,
                    mab,
                    test,
                    optimal_reward,
                    regret_history):

    qlearning = QLearning(num_arms = test.num_arms, 
                          num_states_per_arm = test.num_states_per_arm,
                          init_learning_rate = args.init_learning_rate,
                          discount_factor = test.discount_factor, 
                          temperature_mode = "epsilon-greedy-gittin")

    gittin_history = np.zeros((args.num_runs, args.num_epochs, test.num_arms, test.num_states_per_arm))

    for run in tqdm(range(args.num_runs), unit=" #Run"):
        qlearning.reset()
     
        for epoch in range(args.num_epochs):
            mab.reset(random=True)
            for t in range(args.episode_len):  # just to be consistent with Reinforce algorithm's training time
                cur_states = mab.get_cur_state()
                cur_action, _ = qlearning.get_action(cur_states)
                next_state, cur_reward = mab.step(cur_action)
                qlearning.update(cur_states[cur_action], 
                                 next_state, 
                                 cur_reward, 
                                 cur_action)

            # Store gittins
            for k in range(test.num_arms):
                for n in range(test.num_states_per_arm):
                    gittin_history[run, epoch, k, n] = qlearning.q_table[n, 0, n, k]

            # calculate regret
            regret_history[run][epoch] = calculate_regret(mab, qlearning, args.episode_len, optimal_reward, test.discount_factor)

    if not args.not_show_gittin_plot:
        title = f'Gittins index for num_runs={args.num_runs} and num_epochs={args.num_epochs}'
        savepath = f'{args.savepath}/gittin_plot.png'
        qlearning.qlearning_visualize(gittin_history, title, savepath)

    return
