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
