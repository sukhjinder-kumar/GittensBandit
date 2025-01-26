import numpy as np
from numpy import ndarray
from typing import Annotated
import matplotlib.pyplot as plt

def calculate_optimal_reward(test, mab, episode_len):
    num_runs = 100
    optimal_reward = np.zeros((num_runs), dtype=float)
    for run in range(num_runs):    
        mab.reset(random=False)
        for t in range(episode_len):
            cur_state = mab.get_cur_state()
            action = test.get_highest_rank(cur_state)
            _, reward = mab.step(action)
            optimal_reward[run] += (test.discount_factor**t) * reward

    return np.mean(optimal_reward, axis=0)

def calculate_cumm_reward(strategy, test, mab, episode_len):
    cumm_reward = 0
    mab.reset(random=False)
    for t in range(episode_len):
        cur_state = mab.get_cur_state()
        action, action_probability = strategy.get_action(cur_state)
        _, reward = mab.step(action)
        cumm_reward += (test.discount_factor**t) * reward

    return cumm_reward

def plot_regret_history_average(regret_history_average: Annotated[ndarray, float],
                                strategies: list,
                                title: str,
                                savepath: str):
    num_strategies = regret_history_average.shape[0]
    num_epochs = regret_history_average.shape[1]
    plt.figure(figsize=(10, 6))
    for i in range(regret_history_average.shape[0]):  # loop over all strategies
        plt.plot(range(num_epochs), regret_history_average[i], label=f"{strategies[i].name}")
    plt.xlabel("Epochs")
    plt.ylabel("Regret")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(savepath)
    plt.show()

def plot_cumm_regret_average(cumm_regret_average: Annotated[ndarray, float],
                             strategies: list,
                             title: str,
                             savepath: str):
    num_strategies = cumm_regret_average.shape[0]
    num_epochs = cumm_regret_average.shape[1]
    plt.figure(figsize=(10, 6))
    for i in range(cumm_regret_average.shape[0]):  # loop over all strategies
        plt.plot(range(num_epochs), cumm_regret_average[i], label=f"{strategies[i].name}")
    plt.xlabel("Epochs")
    plt.ylabel("Cumm Regret")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(savepath)
    plt.show()