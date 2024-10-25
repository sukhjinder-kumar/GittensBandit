# Dev Documentation

---

## Setup

1. Create a git repository

2. `git init`, `git remote add origin <ssh>`, `git add . && git commit -m <msg>`, `git push -u origin main`, later only `git push` suffices.

3. `python3 -m venv <env_path>`, `source <env_path>/bin/activate`, `python3 -m pip install --upgrade pip` and finally `pip3 install -r requirements.txt`

> [!WARNING]
> Please note that `torch` is installed for macos not linux.

## Code Rules

1. Use CamelCase for class names only and for anything else use snake_case.

2. Use CamelCase for folder names and snake_case for file names.

3. Base Folder names like "Strategies", "Scripts", etc, all our plural.

4. Use numpy arrays for all array related tasks

    - `np.array(init_array, dtype=int | float | object)` for init known array or `np.zeros(shape, dtype = int | float | object)` for init 2x3 numpy array. Thinking C, each index in axis = 0, i.e. of size 2, has a pointer to an array of size 3.

        - `np.empty(shape, dtype = int | float | object)`

    - To insert just use `a[i][j]`, where first indexing is for first axis, and so on.

    - Note to copy, use `a = b.copy()` where we are trying to copy array b into array a. Make sure they have same size

5. Use numpy for generating random numbers. Use `np.random.choice(space, p, size)` -

    - Where space can be int (than sample from range(int)) or a list. 

    - `p` is weighed probabilities

    - `size` is number of samples

    - if `p` is not present, uniform distribution assumed. If `size` not present, outputs np.int (or float) object otherwise a numpy array. If we use int option and nothing else, outputs is int class. To be safe always add a wrapper for type conversion.

## API interface

For each algorithm, we have the following approx following interface -

```python
class Strategy():
    def __init__(self, name):
        self.name = "Strategy"
    def get_action(self, cur_state):
        # cur_state : np.array((k)) : ith element tells state of ith arm
        return action, action_probability  # \in [0, 1, ... , k-1] which arm to select
    def update_strategy(self, cur_state, action_taken, reward, cummulative_reward=None):
        # action_taken is the output of get_action(cur_state)
        # reward is the reward recieved when we pull action_taken arm
        # cummulative_reward is the discount cummulative reward till the end of episode
        return
```

Some algorithms (like Reinforce) might output a probability distribution for to-select action, which is required for update. Also, similarly for update it requires future knowledge (cummulative reward) to update there internal state.

> [!Note]
> Use `super().__int__(name, self)` to call `init` function of interface class and not rewrite that line and have one hanging instance of self.name. Maybe
