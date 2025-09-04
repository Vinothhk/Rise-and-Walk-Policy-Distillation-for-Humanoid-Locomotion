import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import collections
import time
import wandb # Ensure you have wandb installed: pip install wandb

# Import Stable Baselines3 PPO
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

# --- Configuration and Hyperparameters ---
ENV_NAME = "HumanoidStandup-v5"
LEARNING_RATE = 3e-4 # PPO default in SB3 is often 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2 # This is handled internally by SB3 PPO
PPO_EPOCHS = 10    # This is handled internally by SB3 PPO
MINIBATCH_SIZE = 64 # This is handled internally by SB3 PPO
TIMESTEPS_PER_BATCH = 2048 # This is `n_steps` in SB3 PPO
TOTAL_TIMESTEPS = 5_000_000 # Total environment steps for training
LOG_INTERVAL = 1       # Log every X updates (SB3 logs per training update)
SAVE_INTERVAL = 500_000   # Save model every X timesteps (approximate)

# Set device for GPU acceleration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Weights & Biases Callback for Stable Baselines3 ---
# This custom callback will log metrics to W&B during SB3 training
class WandbCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.last_log_timesteps = 0
        self.start_time = time.time()

    def _on_training_start(self) -> None:
        # Initialize W&B run here if not already done by main script
        if wandb.run is None:
            wandb.init(project="humanoid-drl-teacher", name="ppo_teacher_humanoid_standup_v5_render_sb3", config=self.locals['self'].get_parameters())
            wandb.watch(self.locals['self'].policy, log_freq=1000) # Log policy weights/gradients

    def _on_step(self) -> bool:
        # Log episode rewards and lengths as they complete
        # Note: self.locals['dones'] and self.locals['rewards'] are from the vectorized environment.
        # If using make_vec_env, these will be arrays.
        # If using a single env wrapped by DummyVecEnv, they will be single-element arrays.
        if self.locals['dones'] is not None:
            for i, done in enumerate(self.locals['dones']):
                if done:
                    # Accessing rewards/infos from the vectorized environment
                    # For a single environment, self.locals['rewards'] will be a 1-element array
                    # self.locals['infos'] will be a list of dicts
                    episode_reward = self.locals['rewards'][i]
                    # SB3's info dict for gymnasium environments includes episode stats
                    episode_info = self.locals['infos'][i].get('episode')
                    episode_length = episode_info.get('l', 0) if episode_info else 0
                    
                    wandb.log({
                        "episode_reward": episode_reward,
                        "episode_length": episode_length,
                        "timesteps": self.num_timesteps,
                        "time_elapsed": time.time() - self.start_time
                    }, step=self.num_timesteps)
                    
                    # For terminal logging
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)

        # Log training metrics from SB3's logger to W&B
        # This condition ensures logging happens after a full `n_steps` batch is processed
        if self.num_timesteps - self.last_log_timesteps >= self.model.n_steps:
            self.last_log_timesteps = self.num_timesteps
            
            # Get metrics from SB3's internal logger using name_to_value
            logger_dict = self.logger.name_to_value
            
            # Filter out non-scalar values and log
            log_data = {k: v for k, v in logger_dict.items() if isinstance(v, (int, float))}
            log_data["global_timesteps"] = self.num_timesteps
            wandb.log(log_data, step=self.num_timesteps)

            # Terminal logging for batch summary
            if self.episode_rewards:
                avg_reward_since_last_log = np.mean(self.episode_rewards)
                avg_length_since_last_log = np.mean(self.episode_lengths)
            else:
                avg_reward_since_last_log = 0
                avg_length_since_last_log = 0

            print(f"\n--- Batch Summary (Timesteps: {self.num_timesteps}) ---")
            print(f"Time Elapsed: {time.time() - self.start_time:.2f} seconds")
            print(f"Avg Episode Reward (since last log): {avg_reward_since_last_log:.2f}")
            print(f"Avg Episode Length (since last log): {avg_length_since_last_log:.2f}")
            # Access specific keys from logger_dict for losses
            print(f"Policy Loss: {logger_dict.get('train/policy_loss', 0):.4f}, Value Loss: {logger_dict.get('train/value_loss', 0):.4f}\n")
            
            self.episode_rewards = []
            self.episode_lengths = []

        # Save model checkpoint
        # This condition ensures saving happens approximately every SAVE_INTERVAL timesteps
        # by checking if the current timesteps have crossed a multiple of SAVE_INTERVAL
        # since the last save point, and also ensuring it's not saving every single step.
        if (self.num_timesteps // SAVE_INTERVAL) > (self.last_log_timesteps // SAVE_INTERVAL):
             model_save_path = f"policies/teacher_policy_timesteps_{self.num_timesteps}.zip"
             self.model.save(model_save_path)
             wandb.save(model_save_path)
             print(f"Model saved to {model_save_path}")
        #      model_save_path = f"policies/teacher_policy_timesteps_{self.num_timesteps}.zip"
        #      self.model.save(model_save_path)
        #      wandb.save(model_save_path)
        #      print(f"Model saved to {model_save_path}")

        return True # Continue training

# --- Main Training Loop ---
def train():
    # Define a function to create the environment with rendering
    def make_env_with_render():
        return gym.make(ENV_NAME, render_mode='human')

    # Create the vectorized environment using the factory function
    # This ensures render_mode='human' is passed to the underlying gym.make call
    env = make_vec_env(make_env_with_render, n_envs=1) 

    # Define policy kwargs for the PPO model (e.g., network architecture)
    policy_kwargs = dict(
        net_arch=[dict(pi=[256, 256], vf=[256, 256])] # Two hidden layers of 256 neurons for actor (pi) and critic (vf)
    )

    # Initialize the PPO agent
    model = PPO(
        "MlpPolicy", # Multi-layer Perceptron policy
        env,
        learning_rate=LEARNING_RATE,
        n_steps=TIMESTEPS_PER_BATCH, # Number of steps to run for each environment per update
        batch_size=MINIBATCH_SIZE,   # Minibatch size for PPO's inner loop
        n_epochs=PPO_EPOCHS,         # Number of epochs for PPO's inner loop
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_EPSILON,
        verbose=0, # Set to 1 or 2 for SB3's default logging, 0 to rely on our callback
        device=DEVICE,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./sb3_logs/" # SB3's internal TensorBoard logging
    )
    
    # Configure SB3's logger to use W&B
    new_logger = configure(folder=None, format_strings=["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    # Initialize W&B run (if not already initialized by callback)
    if wandb.run is None:
        wandb.init(project="humanoid-drl-teacher", name="ppo_teacher_humanoid_standup_v5_headless_sb3", config=model.get_parameters())
        wandb.watch(model.policy, log_freq=1000)

    # Train the agent
    print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=WandbCallback(),
        log_interval=LOG_INTERVAL, # This is for SB3's internal logger, not our custom callback
        progress_bar=True # Show a progress bar
    )

    env.close()
    print("\n--- Training Complete ---")
    final_model_path = "policies/teacher_policy_final_sb3.zip"
    model.save(final_model_path)
    wandb.save(final_model_path) # Save final model to W&B
    wandb.finish()

if __name__ == "__main__":
    # Create directories if they don't exist
    import os
    os.makedirs('policies', exist_ok=True)
    os.makedirs('sb3_logs', exist_ok=True) # For SB3's internal tensorboard logs
    
    train()
