import numpy as np
import matplotlib.pyplot as plt

def calculate_regret(mab, strategy, episode_len, optimal_reward, discount_factor):
    num_runs = 20
    cumm_reward = np.zeros((num_runs))
    for run in range(num_runs):
        mab.reset(random=False)
        for t in range(episode_len):
            cur_state = mab.get_cur_state()
            action, action_probability = strategy.get_action(cur_state)
            _, reward = mab.step(action)
            cumm_reward[run] += (discount_factor**t) * reward

    regret = np.mean(cumm_reward, axis=0) - optimal_reward
    return regret

def plot_regret_history_average(regret_history_average,
                                title,
                                savepath):
    plt.plot(regret_history_average)
    plt.xlabel("Epochs")
    plt.ylabel("Regret")
    plt.title(title)
    plt.savefig(savepath)
    plt.show()

def plot_cumm_regret_average(cumm_regret_average,
                             title,
                             savepath):
    plt.plot(cumm_regret_average)
    plt.xlabel("Epochs")
    plt.ylabel("Cumm Regret")
    plt.title(title)
    plt.savefig(savepath)
    plt.show()
