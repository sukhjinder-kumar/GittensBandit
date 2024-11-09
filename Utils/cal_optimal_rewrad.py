import numpy as np

def calculate_optimal_reward(test, mab, episode_len):
    num_runs = 20
    optimal_reward = np.zeros((num_runs))
    for run in range(num_runs):    
        mab.reset(random=False)
        for t in range(episode_len):
            cur_state = mab.get_cur_state()
            action = test.get_highest_rank(cur_state)
            _, reward = mab.step(action)
            optimal_reward[run] += (test.discount_factor**t) * reward

    return np.mean(optimal_reward, axis=0)
