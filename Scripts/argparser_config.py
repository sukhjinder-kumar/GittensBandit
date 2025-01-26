import argparse
from datetime import datetime
import os
import json

def create_folder_if_not_exists(path):
    # Check if the path exists
    if not os.path.exists(path):
        # Create the directory (including any necessary intermediate directories)
        os.makedirs(path)

def get_args():
    # Create the parser
    parser = argparse.ArgumentParser(description="Arguments parser for Gittin Bandit code.")
    
    # Positional arguments (required)
    allowed_tests = ["test1", "test2", "test3", "test4", "test5"]
    parser.add_argument('test_name', type=str, choices=allowed_tests, help="test name")

    # Common Parameters
    parser.add_argument('--num_runs', type=int, help="number of runs to average result over", default=100)
    parser.add_argument('--num_epochs', type=int, help="number of epoch", default=1000)
    parser.add_argument('--episode_len', type=int, help="length of an episode within an epoch", default=20)
    parser.add_argument('--not_show_regret_average_plot', action='store_true',
                        help="include to not show the average regret plot")
    parser.add_argument('--not_show_cumm_regret_average_plot', action='store_true',
                        help="include to not show average cumm regret plot")

    # Q-Learning Algorithm
    parser.add_argument('--qlearning_init_learning_rate', type=float, 
                        help="init learning rate for qlearning algorithm", default=0.05)
    parser.add_argument('--qlearning_tau', type=float, 
                        help="tau for QLearning", default=300)
    parser.add_argument('--qlearning_not_show_gittin_plot', action='store_true', 
                        help="include to not show the gittins plot")

    allowed_schedule = ["Boltzmann", "epsilon-greedy"]
    parser.add_argument('--qlearning_schedule', type=str, choices=allowed_schedule,
                        help="schedule for qlearning algorithm", default="epsilon-greedy")
    parser.add_argument('--qlearning_max_temperature', type=float, 
                        help="max temperature for qlearning algorithm", default=200)
    parser.add_argument('--qlearning_min_temperature', type=float, 
                        help="min temperature for qlearning algorithm", default=0.5)
    parser.add_argument('--qlearning_beta', type=float, 
                        help="beta for Boltzmann schedule for qlearning algorithm", default=0.992)
    parser.add_argument('--qlearning_epsilon_greedy', type=float,
                        help="epsilon for epsilon greedy schedule", default=0.1)

    # Reinforce Algorithm
    parser.add_argument('--reinforce_learning_rate', type=float,
                        help="learning rate for reinforce algorithm", default=0.001)
    allowed_schedules = ["none", "linear"]
    parser.add_argument('--reinforce_schedule', type=str, choices=allowed_schedules,
                        help="schedule for temperature in reinforce algorithm", default="linear")

    parser.add_argument('--reinforce_max_temperature', type=float, 
                        help="max temperature for reinforce algorithm", default=200)
    parser.add_argument('--reinforce_min_temperature', type=float, 
                        help="min temperature for reinforce algorithm", default=0.5)
    parser.add_argument('--reinforce_beta', type=float, 
                        help="beta for Boltzmann schedule for reinforce algorithm", default=0.992)
    parser.add_argument('--reinforce_constant_temperature', type=float, 
                        help="constant temperature for reinforce algorithm", default=1)

    parser.add_argument('--reinforce_not_show_preference_plot', action='store_true', 
                        help="include to not show the preference plot")


    # Save path
    formatted_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    default_savepath = f"Results/{formatted_time}"
    parser.add_argument('--savepath', type=str, 
                        help="what path to store the plots and parameter value", default=default_savepath)
    args = parser.parse_args()  # .parse_args() closes the args object, no more additions can be  made
    create_folder_if_not_exists(args.savepath)

    # Store all parameters in a file in the savepath directory
    args_dict = vars(args)  # Convert argparse Namespace to a dictionary
    with open(f"{args.savepath}/args.json", "w") as f:  # Save to JSON file
        json.dump(args_dict, f, indent=4)

    # Parse and return arguments
    return args