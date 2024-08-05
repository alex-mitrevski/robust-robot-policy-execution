"""
Traning script for the PPO model trained to reach the door.

Author: Bharath Santhanam
Email: bharathsanthanamdev@gmail.com
Organization: Hochschule Bonn-Rhein-Sieg


Description:
This script trains the PPO model
The script loads the configuration file, creates the environment, loads the model, and trains the policy.


References:
This script is based on:
The entire structure of this script is adapted from https://github.com/NJ-2020-thesis/PyRep/blob/feature/examples/vmp/vmp_trainer.py. Specifc lines referred are mentioned in the code.
Referred to the stable-baselines3 documentations for specific in-built functions: https://stable-baselines3.readthedocs.io/en/master/

"""


import os
import time
import uuid
import argparse
from os.path import abspath, dirname, join
from timeit import default_timer as timer

import yaml
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from simulation_env.door_env_with_joints import ReachDoor


def load_config(config_path):
    with open(config_path, "r") as config_file:
        return yaml.safe_load(config_file)


def create_environment():
    """
    generate the environment, put it under Monitor wrapper
    """
    env = ReachDoor()
    check_env(env)
    return Monitor(env)


def create_model(env, config):
    return PPO(
        "MultiInputPolicy",
        env=env,
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"],
        ent_coef=config["ent_coef"],
        n_epochs=config["n_epochs"],
        learning_rate=config["learning_rate"],
        verbose=1,
        tensorboard_log=join(config["model_save_path"], "log"),
    )


def train_model(model, env, config, checkpoint_callback):
    start_time = timer()
    model.learn(total_timesteps=config["total_timesteps"], callback=checkpoint_callback)
    end_time = timer()
    return end_time - start_time


def main(config_path):
    config = load_config(config_path)
    env = create_environment()

    #adapted from https://github.com/NJ-2020-thesis/PyRep/blob/6f02f0b347654a4bf3fd561a044e00bf85754ba6/examples/vmp/vmp_trainer.py#L87
    checkpoint_callback = CheckpointCallback(
        save_freq=config["checkpoint_save_freq"],
        save_path=join(config["model_save_path"], f"{str(uuid.uuid4())[:5]}"),
        name_prefix="gpu",
    )

    model = create_model(env, config)
    print(model.policy)

    training_time = train_model(model, env, config, checkpoint_callback)
    print(f"Training time: {training_time:.2f} seconds")

    env.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VMP Trainer")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_ppo.yaml",
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    main(args.config)
