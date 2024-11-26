#!/bin/bash

# QLearning + non-homogeneous
echo "Starting QLearning test1 with Boltzmann temperature mode"
python3 -m Scripts.main QLearning test1 --num_runs=10 --num_epochs=20 \
  --episode_len=20 --temperature=200 --temperature_mode="Boltzmann"
echo "Finished QLearning test1 with Boltzmann temperature mode"

echo "Starting QLearning test1 with epsilon-greedy-gittin temperature mode"
python3 -m Scripts.main QLearning test1 --num_runs=10 --num_epochs=20 \
  --episode_len=20 --temperature=200 --temperature_mode="epsilon-greedy-gittin"
echo "Finished QLearning test1 with epsilon-greedy-gittin temperature mode"

# QLearning + homogeneous
echo "Starting QLearning test2 with Boltzmann temperature mode"
python3 -m Scripts.main QLearning test2 --num_runs=10 --num_epochs=20 \
  --episode_len=20 --temperature=200 --temperature_mode="Boltzmann"
echo "Finished QLearning test2 with Boltzmann temperature mode"

echo "Starting QLearning test2 with epsilon-greedy-gittin temperature mode"
python3 -m Scripts.main QLearning test2 --num_runs=10 --num_epochs=20 \
  --episode_len=20 --temperature=200 --temperature_mode="epsilon-greedy-gittin"
echo "Finished QLearning test2 with epsilon-greedy-gittin temperature mode"

# Reinforce + non-homogeneous 
echo "Starting Reinforce test1 with linear schedule"
python3 -m Scripts.main Reinforce test1 --num_runs=10 --num_epochs=20 \
  --episode_len=20 --temperature=200 --schedule="linear"
echo "Finished Reinforce test1 with linear schedule"

echo "Starting Reinforce test1 with none schedule"
python3 -m Scripts.main Reinforce test1 --num_runs=10 --num_epochs=20 \
  --episode_len=20 --temperature=200 --schedule="none"
echo "Finished Reinforce test1 with none schedule"

# Reinforce + homogeneous 
echo "Starting Reinforce test2 with linear schedule"
python3 -m Scripts.main Reinforce test2 --num_runs=10 --num_epochs=20 \
  --episode_len=20 --temperature=200 --schedule="linear"
echo "Finished Reinforce test2 with linear schedule"

echo "Starting Reinforce test2 with none schedule"
python3 -m Scripts.main Reinforce test2 --num_runs=10 --num_epochs=20 \
  --episode_len=20 --temperature=200 --schedule="none"
echo "Finished Reinforce test2 with none schedule"

# NeuralReinforce + non-homogeneous 
echo "Starting NeuralReinforce test1 with linear schedule"
python3 -m Scripts.main NeuralReinforce test1 --num_runs=2 --num_epochs=2 \
  --episode_len=20 --temperature=200 --schedule="linear"
echo "Finished NeuralReinforce test1 with linear schedule"

# NeuralReinforce + homogeneous 
echo "Starting NeuralReinforce test2 with linear schedule"
python3 -m Scripts.main NeuralReinforce test2 --num_runs=2 --num_epochs=2 \
  --episode_len=20 --temperature=200 --schedule="linear"
echo "Finished NeuralReinforce test2 with linear schedule"
