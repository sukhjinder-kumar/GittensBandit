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
    allowed_strategies = ["QLearning", "Reinforce", "NeuralReinforce"]
    allowed_tests = ["test1", "test2", "test3"]
    parser.add_argument('strategy_name', type=str, choices=allowed_strategies, help="Strategy name")
    parser.add_argument('test_name', type=str, choices=allowed_tests, help="test name")

    # Common Parameters
    parser.add_argument('--num_runs', type=int, help="number of runs to average result over", default=100)
    parser.add_argument('--num_epochs', type=int, help="number of epoch", default=1000)
    parser.add_argument('--episode_len', type=int, help="length of an episode within an epoch", default=20)

    # Conditional Parsing
    args, remaining_argv = parser.parse_known_args()

    # Q-Learning Algorithm
    if args.strategy_name == "QLearning":
        parser.add_argument('--temperature_mode', type=str, 
                            help="temperature mode for qlearning algorithm", default="epsilon-greedy-gittin")
        parser.add_argument('--init_learning_rate', type=float, 
                            help="init learning rate for qlearning algorithm", default=0.05)
        parser.add_argument('--not_show_gittin_plot', action='store_true', 
                            help="include to not show the gittins plot")

    # Reinforce Algorithm
    if args.strategy_name == "Reinforce":
        parser.add_argument('--temperature', type=int, 
                            help="temperature for reinforce algorithm", default=1)
        parser.add_argument('--schedule', type=str, 
                            help="schedule for temperature in reinforce algorithm", default="linear")
        parser.add_argument('--learning_rate', type=float, 
                            help="learning rate for reinforce algorithm", default=0.001)
        parser.add_argument('--not_show_preference_plot', action='store_true', 
                            help="include to not show the preference plot")

    parser.add_argument('--not_show_regret_average_plot', action='store_true',
                        help="include to not show the average regret plot")
    parser.add_argument('--not_show_cumm_regret_average_plot', action='store_true',
                        help="include to not show average cumm regret plot")

    # Save path
    formatted_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    default_savepath = f"Results/{args.strategy_name}/{formatted_time}"
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
