import os
import sys
import random

import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from snake_game_custom_wrapper_cnn import SnakeEnv

if torch.backends.mps.is_available():
    NUM_ENV = 32 * 2
else:
    NUM_ENV = 32
LOG_DIR = "logs"

os.makedirs(LOG_DIR, exist_ok=True)

# Linear scheduler
def linear_schedule(initial_value, final_value=0.0):

    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0)

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler

def make_env(seed=0):
    """
    创建并配置环境的函数工厂。

    该函数的主要作用是根据给定的种子生成一个特定的环境配置，包括随机数种子的设置、动作掩码的添加和监控器的包装。

    参数:
    - seed (int): 用于环境和随机数生成器的种子，以确保实验的可重复性。

    返回:
    - function: 一个无参数函数，当调用时，返回一个配置好的环境实例。
    """

    def _init():
        """
        初始化和包装环境的内部函数。

        该函数按顺序执行以下操作：
        1. 创建基础蛇类环境。
        2. 应用动作掩码，限制非法动作。
        3. 使用监控器包装环境，以跟踪奖励和完成情况。
        4. 设置环境的随机数种子。

        返回:
        - env: 配置好的环境实例。
        """
        env = SnakeEnv(seed=seed)
        print(f"Original action space: {env.action_space}")  # 调试输出
        env = ActionMasker(env, SnakeEnv.get_action_mask)
        print(f"Action space after ActionMasker: {env.action_space}")  # 调试输出
        env = Monitor(env)
        return env

    return _init


def main():

    # Generate a list of random seeds for each environment.
    seed_set = set()
    while len(seed_set) < NUM_ENV:
        seed_set.add(random.randint(0, 1e9))

    # Create the Snake environment.
    env = SubprocVecEnv([make_env(seed=s) for s in seed_set])

    if torch.backends.mps.is_available():
        lr_schedule = linear_schedule(5e-4, 2.5e-6)
        clip_range_schedule = linear_schedule(0.150, 0.025)
        # Instantiate a PPO agent using MPS (Metal Performance Shaders).
        model = MaskablePPO(
            "CnnPolicy",
            env,
            device="mps",
            verbose=1,
            n_steps=2048,
            batch_size=512*8,
            n_epochs=4,
            gamma=0.94,
            learning_rate=lr_schedule,
            clip_range=clip_range_schedule,
            tensorboard_log=LOG_DIR
        )
    else:
        lr_schedule = linear_schedule(2.5e-4, 2.5e-6)
        clip_range_schedule = linear_schedule(0.150, 0.025)
        # Instantiate a PPO agent using CUDA.
        model = MaskablePPO(
            "CnnPolicy",
            env,
            device="cuda",
            verbose=1,
            n_steps=2048,
            batch_size=512,
            n_epochs=4,
            gamma=0.94,
            learning_rate=lr_schedule,
            clip_range=clip_range_schedule,
            tensorboard_log=LOG_DIR
        )

    # Set the save directory
    if torch.backends.mps.is_available():
        save_dir = "trained_models_cnn_mps"
    else:
        save_dir = "trained_models_cnn"
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_interval = 15625 # checkpoint_interval * num_envs = total_steps_per_checkpoint
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path=save_dir, name_prefix="ppo_snake")

    # Writing the training logs from stdout to a file
    original_stdout = sys.stdout
    log_file_path = os.path.join(save_dir, "training_log.txt")
    with open(log_file_path, 'w') as log_file:
        sys.stdout = log_file

        model.learn(
            total_timesteps=int(100000000),
            callback=[checkpoint_callback]
        )
        env.close()

    # Restore stdout
    sys.stdout = original_stdout

    # Save the final model
    model.save(os.path.join(save_dir, "ppo_snake_final.zip"))

if __name__ == "__main__":
    main()
