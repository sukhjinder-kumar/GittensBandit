#!/bin/bash

# Test 2

num_runs=100
num_epochs=5000
episode_len=20
temperature=200

# Function to convert seconds to MM:SS format
convert_to_mmss() {
  minutes=$(( $1 / 60 ))
  seconds=$(( $1 % 60 ))
  printf "%02d:%02d\n" $minutes $seconds
}

# Function to kill all pids on interrupt
cleanup() {
  echo "Interrupt received! Killing all processes..."
  kill $pid1 $pid2 $pid3
  exit 1
}

# Trap SIGINT (Ctrl+C) to call cleanup function
trap cleanup SIGINT

# QLearning
echo "Starting QLearning test1 with epsilon-greedy-gittin temperature mode"
start_time1=$(date +%s)  # Record start time

sleep 10
python3 -m Scripts.main QLearning test1 --num_runs=${num_runs} --num_epochs=${num_epochs} \
  --episode_len=${episode_len} --temperature=${temperature} --temperature_mode="epsilon-greedy-gittin" &

pid1=$!  # Capture the process ID of the first command

# Reinforce + linear schedule
echo "Starting Reinforce test1 with linear schedule"
start_time2=$(date +%s)  # Record start time

sleep 10
python3 -m Scripts.main Reinforce test1 --num_runs=${num_runs} --num_epochs=${num_epochs} \
  --episode_len=${episode_len} --temperature=${temperature} --schedule="linear" &

pid2=$!  # Capture the process ID of the second command

# Reinforce + none schedule
echo "Starting Reinforce test1 with none schedule"
start_time3=$(date +%s)  # Record start time

sleep 10
python3 -m Scripts.main Reinforce test1 --num_runs=${num_runs} --num_epochs=${num_epochs} \
  --episode_len=${episode_len} --temperature=1 --schedule="none" &

pid3=$!  # Capture the process ID of the third command

# Wait for the first command to finish
wait $pid1
end_time1=$(date +%s)  # Record end time
duration1=$((end_time1 - start_time1))  # Calculate duration in seconds
duration1_formatted=$(convert_to_mmss $duration1)  # Convert to MM:SS
echo "Finished QLearning test1 with epsilon-greedy-gittin temperature mode in $duration1_formatted"

# Wait for the second command to finish
wait $pid2
end_time2=$(date +%s)  # Record end time
duration2=$((end_time2 - start_time2))  # Calculate duration in seconds
duration2_formatted=$(convert_to_mmss $duration2)  # Convert to MM:SS
echo "Finished Reinforce test1 with linear schedule in $duration2_formatted"

# Wait for the third command to finish
wait $pid3
end_time3=$(date +%s)  # Record end time
duration3=$((end_time3 - start_time3))  # Calculate duration in seconds
duration3_formatted=$(convert_to_mmss $duration3)  # Convert to MM:SS
echo "Finished Reinforce test1 with none schedule in $duration3_formatted"
