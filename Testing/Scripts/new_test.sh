#!/bin/bash

num_runs=2
num_epochs=10
episode_len=20

# QLearning
qlearning_init_learning_rate=0.05
qlearning_tau=300
qlearning_schedule="Boltzmann"  # Or "epsilon-greedy"
qlearning_max_temperature=200
qlearning_min_temperature=0.5
qlearning_beta=0.992
# qlearning_epsilon_greedy=0.1

# Reinforce
reinforce_learning_rate=0.001
reinforce_schedule="linear"  # or "none"
reinforce_max_temperature=200
reinforce_min_temperature=0.5
reinforce_beta=0.992
# reinforce_constant_temperature=1

# Function to convert seconds to MM:SS format
convert_to_mmss() {
  minutes=$(( $1 / 60 ))
  seconds=$(( $1 % 60 ))
  printf "%02d:%02d\n" $minutes $seconds
}

echo "Starting" 
start_time=$(date +%s)  # Record start time

# --qlearning_epsilon_greedy=${qlearning_epsilon_greedy} \
# --reinforce_constant_temperature=${reinforce_constant_temperature} \

python3 -m Scripts.main test3 \
        --num_runs=${num_runs} \
        --num_epochs=${num_epochs} \
        --episode_len=${episode_len} \
        --qlearning_init_learning_rate=${qlearning_init_learning_rate} \
        --qlearning_tau=${qlearning_tau} \
        --qlearning_schedule=${qlearning_schedule} \
        --qlearning_max_temperature=${qlearning_max_temperature} \
        --qlearning_min_temperature=${qlearning_min_temperature} \
        --qlearning_beta=${qlearning_beta} \
        --reinforce_learning_rate=${reinforce_learning_rate} \
        --reinforce_schedule=${reinforce_schedule} \
        --reinforce_max_temperature=${reinforce_max_temperature} \
        --reinforce_min_temperature=${reinforce_min_temperature} \
        --reinforce_beta=${reinforce_beta} \

end_time=$(date +%s)  # Record end time
duration=$((end_time - start_time))  # Calculate duration in seconds
duration_formatted=$(convert_to_mmss $duration)  # Convert to MM:SS
echo "Finished training in $duration_formatted"