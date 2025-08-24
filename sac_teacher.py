# File: train_optimal.py

import gymnasium as gym
import time
import os
import multiprocessing
import numpy as np
import wandb

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from wandb.integration.sb3 import WandbCallback

# --- Configuration ---
# config = {
#     "policy_type": "MlpPolicy",
#     "total_timesteps": 3_000_000, # A good target for Humanoid
#     "env_name": "HumanoidStandup-v5",
#     "n_envs": multiprocessing.cpu_count(),
#     "learning_rate": 3e-4, # Standard for SAC
#     "buffer_size": 1_000_000,
#     "batch_size": 256,
#     "gamma": 0.99,
#     "tau": 0.005,
#     "ent_coef": 'auto',
#     "learning_starts": 10000,
#     "policy_kwargs": dict(net_arch=[256, 256]),https://gymnasium.farama.org/environments/mujoco/humanoid_standup/
#     "log_interval_steps": 2048 * 4, # Print console logs every ~8k steps
# }

# --- Configuration ---
config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 3_000_000,
    "env_name": "HumanoidStandup-v5",
    
    # FIX 1: Reduce the number of parallel environments to a fixed, smaller number
    "n_envs": 4, # Instead of multiprocessing.cpu_count()
    
    "learning_rate": 3e-4,
    
    # FIX 2: Reduce the buffer size to lower RAM usage
    "buffer_size": 250_000, # Drastically reduced from 1,000,000
    
    "batch_size": 256,
    "gamma": 0.99,
    "tau": 0.005,
    "ent_coef": 'auto',
    "learning_starts": 10000,
    "policy_kwargs": dict(net_arch=[256, 256]),
    "log_interval_steps": 2048 * 4,
}

# --- Custom Callback for Console Logging ---
class ConsoleLogCallback(BaseCallback):
    """
    A custom callback that prints a clean, formatted summary to the console.
    """
    def __init__(self, log_interval, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.last_log_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_log_step >= self.log_interval:
            self.last_log_step = self.num_timesteps
            
            # --- FIX IS HERE ---
            # Get latest logged values from the logger's dictionary
            # Use .get(key, 0) to avoid errors if the key is not yet available
            ep_rew_mean = self.logger.name_to_value.get('rollout/ep_rew_mean', 0)
            ep_len_mean = self.logger.name_to_value.get('rollout/ep_len_mean', 0)
            actor_loss = self.logger.name_to_value.get('train/actor_loss', 0)
            critic_loss = self.logger.name_to_value.get('train/critic_loss', 0)
            # --- END OF FIX ---

            # print(f"Timesteps: {self.num_timesteps:<8} | "
            #       f"Avg Reward: {ep_rew_mean:<8.2f} | "
            #       f"Avg Ep Length: {ep_len_mean:<8.2f} | "
            #       f"Actor Loss: {actor_loss:<8.4f} | "
            #       f"Critic Loss: {critic_loss:<8.4f}")
        return True

# --- Main Training Function ---
def train():
    PROJECT_NAME = "HumanoidStandup-Optimal-SAC"
    RUN_NAME = f"SAC_{config['env_name']}_{int(time.time())}"
    
    # Initialize W&B run
    run = wandb.init(
        project=PROJECT_NAME,
        name=RUN_NAME,
        config=config,
        sync_tensorboard=True, # Key for automatic logging
        monitor_gym=True,
        save_code=True,
    )

    # Create vectorized environment
    env = make_vec_env(
        config["env_name"], 
        n_envs=config["n_envs"], 
        vec_env_cls=SubprocVecEnv
    )

    save_path = f"models/{run.id}"
    os.makedirs(save_path, exist_ok=True)

    # Create callbacks
    # 1. W&B Callback for comprehensive logging
    wandb_callback = WandbCallback(
        model_save_path=f"models/{run.id}",
        verbose=0,
    )
    
    # 2. Evaluation callback to save the best model
    eval_env = gym.make(config["env_name"])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        eval_freq=max(10000 // config["n_envs"], 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )
    
    # 3. Custom callback for clean console logs
    console_callback = ConsoleLogCallback(log_interval=config["log_interval_steps"])

    # Instantiate the SAC model
    model = SAC(
        policy=config["policy_type"],
        env=env,
        learning_rate=config["learning_rate"],
        buffer_size=config["buffer_size"],
        batch_size=config["batch_size"],
        gamma=config["gamma"],
        tau=config["tau"],
        ent_coef=config["ent_coef"],
        learning_starts=config["learning_starts"],
        policy_kwargs=config["policy_kwargs"],
        verbose=1, # Set to 0 to disable SB3's default logging and use our custom one
        tensorboard_log=f"runs/{run.id}",
    )

    # Train the model with all three callbacks
    print("🚀 Starting optimal training...")
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=[wandb_callback, eval_callback, console_callback],
        progress_bar=True,  # Show progress bar in console
    )
    
    run.finish()
    print("✅ Training finished!")

if __name__ == "__main__":
    train()