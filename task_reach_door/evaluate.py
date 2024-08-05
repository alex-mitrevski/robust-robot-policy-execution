"""
Evaluation script for the PPO model trained to reach the door.

Author: Bharath Santhanam
Email: bharathsanthanamdev@gmail.com
Organization: Hochschule Bonn-Rhein-Sieg


Description:
This script evaluates the PPO model trained to reach the door.
The script loads the configuration file, creates the environment, loads the model, and evaluates the policy.


References:
This script is based on:
The entire structure of this script is adapted from https://github.com/NJ-2020-thesis/PyRep/blob/feature/examples/vmp/vmp_evaluator.py . Specifc lines referred are mentioned in the code.
Referred to the stable-baselines3 documentations for specific in-built functions: https://stable-baselines3.readthedocs.io/en/master/

"""

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from simulation_env.door_env_with_joints import ReachDoor


def load_config(config_path):
    with open(config_path, "r") as config_file:
        return yaml.safe_load(config_file)


def main():
    # Load configuration
    config = load_config("config/config_ppo.yaml")
    eval_config = config["evaluation"]

    # Create and check environment
    env = ReachDoor()
    check_env(env)
    env = Monitor(env)

    # Load model
    model = PPO.load(eval_config["model_path"], env=env, verbose=eval_config["verbose"])

    # This function is adapted from the implementation of
    # https://github.com/NJ-2020-thesis/PyRep/blob/6f02f0b347654a4bf3fd561a044e00bf85754ba6/examples/vmp/vmp_evaluator.py#L44
    # Evaluate policy
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        deterministic=eval_config["deterministic"],
        n_eval_episodes=eval_config["n_eval_episodes"],
        return_episode_rewards=False,
    )

    print("====>", "mean_reward:", mean_reward, "std_reward:", std_reward)

    # Cleanup
    env.close()
    env.shutdown()


if __name__ == "__main__":
    main()
