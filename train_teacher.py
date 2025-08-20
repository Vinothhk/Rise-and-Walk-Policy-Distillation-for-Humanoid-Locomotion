# import gymnasium as gym
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.distributions import Normal
# import numpy as np
# import collections
# import time
# import wandb # Ensure you have wandb installed: pip install wandb

# # --- Configuration and Hyperparameters ---
# ENV_NAME = "HumanoidStandup-v5" # Changed to v5
# LEARNING_RATE = 3e-4
# GAMMA = 0.99
# GAE_LAMBDA = 0.95
# CLIP_EPSILON = 0.2
# PPO_EPOCHS = 10        # Number of times to iterate over collected data
# MINIBATCH_SIZE = 64
# TIMESTEPS_PER_BATCH = 2048 # How many environment steps to collect before a PPO update
# TOTAL_TIMESTEPS = 5_000_000 # Total environment steps for training
# LOG_INTERVAL = 10       # Log every X batches
# SAVE_INTERVAL = 50      # Save model every X batches

# # Set device for GPU acceleration
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {DEVICE}")

# # --- Initialize Weights & Biases ---
# wandb.init(project="humanoid-drl-teacher", name="ppo_teacher_humanoid_standup_v5_render", config={ # Updated name for clarity
#     "env_name": ENV_NAME,
#     "learning_rate": LEARNING_RATE,
#     "gamma": GAMMA,
#     "gae_lambda": GAE_LAMBDA,
#     "clip_epsilon": CLIP_EPSILON,
#     "ppo_epochs": PPO_EPOCHS,
#     "minibatch_size": MINIBATCH_SIZE,
#     "timesteps_per_batch": TIMESTEPS_PER_BATCH,
#     "total_timesteps": TOTAL_TIMESTEPS,
#     "device": str(DEVICE)
# })

# # --- Network Definitions (models/actor_critic.py content) ---
# class Actor(nn.Module):
#     def __init__(self, obs_dim, action_dim, hidden_dim=256):
#         super(Actor, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(obs_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, action_dim) # Output mean of action distribution
#         )
#         # We use a learnable log_std for exploration, initialized to zeros (std=1)
#         self.log_std = nn.Parameter(torch.zeros(action_dim))

#     def forward(self, x):
#         action_mean = self.net(x)
#         action_std = torch.exp(self.log_std) # Standard deviation (must be positive)
#         return action_mean, action_std

# class Critic(nn.Module):
#     def __init__(self, obs_dim, hidden_dim=256):
#         super(Critic, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(obs_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1) # Output value estimate
#         )

#     def forward(self, x):
#         return self.net(x)

# # --- PPO Agent Implementation ---
# class PPOAgent:
#     def __init__(self, obs_dim, action_dim):
#         self.actor = Actor(obs_dim, action_dim).to(DEVICE)
#         self.critic = Critic(obs_dim).to(DEVICE)
#         self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
#         self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)

#         # Log models to W&B
#         wandb.watch(self.actor, log_freq=SAVE_INTERVAL)
#         wandb.watch(self.critic, log_freq=SAVE_INTERVAL)

#     def get_action(self, obs):
#         # Convert observation to PyTorch tensor and move to device
#         obs_tensor = torch.FloatTensor(obs).to(DEVICE)
        
#         # Get action distribution parameters from actor
#         mean, std = self.actor(obs_tensor)
        
#         # Create a normal distribution
#         dist = Normal(mean, std)
        
#         # Sample an action and get its log probability
#         action = dist.sample()
#         log_prob = dist.log_prob(action).sum(axis=-1) # Sum log_probs for continuous action vector
        
#         # Get value estimate from critic
#         value = self.critic(obs_tensor)
        
#         # Detach tensors from computation graph before converting to numpy
#         return action.detach().cpu().numpy(), log_prob.detach().cpu().numpy(), value.detach().cpu().numpy()

#     def update(self, rollouts):
#         # Extract data from rollouts
#         # Ensure all data is on CPU before converting to numpy array
#         obs_batch = torch.FloatTensor(np.array(rollouts['observations'])).to(DEVICE)
#         actions_batch = torch.FloatTensor(np.array(rollouts['actions'])).to(DEVICE)
#         log_probs_old_batch = torch.FloatTensor(np.array(rollouts['log_probs'])).to(DEVICE)
#         rewards_batch = torch.FloatTensor(np.array(rollouts['rewards'])).to(DEVICE)
#         values_batch = torch.FloatTensor(np.array(rollouts['values'])).to(DEVICE)
#         dones_batch = torch.FloatTensor(np.array(rollouts['dones'])).to(DEVICE)
        
#         # Compute advantages (GAE) and returns
#         returns = []
#         advantages = []
#         last_advantage = 0.0 # Initialize as float
#         for i in reversed(range(len(rewards_batch))):
#             current_reward = rewards_batch[i].item() # Get scalar float
#             current_value = values_batch[i].item()   # Get scalar float
#             current_done = dones_batch[i].item()     # Get scalar float

#             if i == len(rewards_batch) - 1: # Last step
#                 next_value = 0.0 # Ensure it's a float
#             else:
#                 # Ensure next_value is a scalar float
#                 next_value = values_batch[i+1].item() * (1 - dones_batch[i+1].item()) 

#             delta = current_reward + GAMMA * next_value - current_value
#             advantage = delta + GAMMA * GAE_LAMBDA * last_advantage * (1 - current_done)
            
#             # Append scalar floats to lists
#             advantages.insert(0, advantage) 
#             returns.insert(0, advantage + current_value)
#             last_advantage = advantage

#         # Convert lists of Python floats to numpy arrays, then to GPU tensors
#         advantages_batch = torch.FloatTensor(np.array(advantages)).to(DEVICE)
#         returns_batch = torch.FloatTensor(np.array(returns)).to(DEVICE)

#         # Normalize advantages
#         advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

#         # Convert to flat batches for minibatching
#         total_samples = len(obs_batch)
#         indices = np.arange(total_samples)

#         for _ in range(PPO_EPOCHS):
#             np.random.shuffle(indices)
#             for start_idx in range(0, total_samples, MINIBATCH_SIZE):
#                 end_idx = start_idx + MINIBATCH_SIZE
#                 batch_indices = indices[start_idx:end_idx]

#                 # Get mini-batch data
#                 mb_obs = obs_batch[batch_indices]
#                 mb_actions = actions_batch[batch_indices]
#                 mb_log_probs_old = log_probs_old_batch[batch_indices]
#                 mb_returns = returns_batch[batch_indices]
#                 mb_advantages = advantages_batch[batch_indices]

#                 # Calculate new log probabilities and values
#                 mean, std = self.actor(mb_obs)
#                 dist = Normal(mean, std)
#                 log_probs_new = dist.log_prob(mb_actions).sum(axis=-1)
#                 values_new = self.critic(mb_obs).squeeze(-1) # Remove last dimension

#                 # Actor loss (clipped PPO objective)
#                 ratio = torch.exp(log_probs_new - mb_log_probs_old)
#                 surr1 = ratio * mb_advantages
#                 surr2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * mb_advantages
#                 actor_loss = -torch.min(surr1, surr2).mean()

#                 # Critic loss (MSE)
#                 critic_loss = 0.5 * (values_new - mb_returns).pow(2).mean()

#                 # Update actor
#                 self.actor_optimizer.zero_grad()
#                 actor_loss.backward()
#                 self.actor_optimizer.step()

#                 # Update critic
#                 self.critic_optimizer.zero_grad()
#                 critic_loss.backward()
#                 self.critic_optimizer.step()
        
#         return actor_loss.item(), critic_loss.item()

#     def save_model(self, path):
#         torch.save({
#             'actor_state_dict': self.actor.state_dict(),
#             'critic_state_dict': self.critic.state_dict(),
#             'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
#             'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
#         }, path)
#         print(f"Model saved to {path}")

# # --- Main Training Loop ---
# def train():
#     # Initialize environment with render_mode='human'
#     env = gym.make(ENV_NAME, render_mode='human') # Added render_mode
#     obs_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.shape[0]

#     agent = PPOAgent(obs_dim, action_dim)

#     current_timesteps = 0
#     batch_num = 0
#     start_time = time.time()

#     # Variables to track episode stats for terminal logging
#     terminal_episode_rewards = []
#     terminal_episode_lengths = []

#     while current_timesteps < TOTAL_TIMESTEPS:
#         rollouts = collections.defaultdict(list)
#         obs, _ = env.reset()
#         done = False
#         episode_reward = 0
#         episode_len = 0

#         # Collect data for one batch
#         for _ in range(TIMESTEPS_PER_BATCH):
#             action, log_prob, value = agent.get_action(obs)
#             next_obs, reward, terminated, truncated, _ = env.step(action)
#             done = terminated or truncated

#             rollouts['observations'].append(obs)
#             rollouts['actions'].append(action)
#             rollouts['log_probs'].append(log_prob)
#             rollouts['rewards'].append(reward)
#             rollouts['values'].append(value)
#             rollouts['dones'].append(float(done))

#             obs = next_obs
#             episode_reward += reward
#             episode_len += 1
#             current_timesteps += 1

#             # Render the environment
#             env.render() # Added render call

#             if done:
#                 # Log episode stats to W&B
#                 wandb.log({
#                     "episode_reward": episode_reward,
#                     "episode_length": episode_len,
#                     "timesteps": current_timesteps,
#                     "time_elapsed": time.time() - start_time
#                 })
                
#                 # Log episode stats to terminal
#                 terminal_episode_rewards.append(episode_reward)
#                 terminal_episode_lengths.append(episode_len)
#                 print(f"--- Episode Finished --- Timesteps: {current_timesteps}, "
#                       f"Reward: {episode_reward:.2f}, Length: {episode_len}")

#                 # Reset for next episode within the batch
#                 obs, _ = env.reset()
#                 episode_reward = 0
#                 episode_len = 0
        
#         # Update policy using the collected batch
#         actor_loss, critic_loss = agent.update(rollouts)
#         batch_num += 1

#         # Log batch-level stats to terminal and W&B
#         if batch_num % LOG_INTERVAL == 0:
#             # Calculate average from terminal_episode_rewards collected since last log
#             if terminal_episode_rewards: # Avoid division by zero if no episodes finished in interval
#                 avg_reward_since_last_log = np.mean(terminal_episode_rewards)
#                 avg_length_since_last_log = np.mean(terminal_episode_lengths)
#             else:
#                 avg_reward_since_last_log = 0
#                 avg_length_since_last_log = 0

#             print(f"\n--- Batch Summary ({batch_num}) ---")
#             print(f"Total Timesteps: {current_timesteps}/{TOTAL_TIMESTEPS}")
#             print(f"Time Elapsed: {time.time() - start_time:.2f} seconds")
#             print(f"Avg Episode Reward (since last log): {avg_reward_since_last_log:.2f}")
#             print(f"Avg Episode Length (since last log): {avg_length_since_last_log:.2f}")
#             print(f"Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}\n")
            
#             wandb.log({
#                 "actor_loss": actor_loss,
#                 "critic_loss": critic_loss,
#                 "avg_episode_reward_batch": avg_reward_since_last_log,
#                 "avg_episode_length_batch": avg_length_since_last_log,
#                 "global_timesteps": current_timesteps
#             }, step=current_timesteps)

#             # Clear episode stats for the next interval
#             terminal_episode_rewards = []
#             terminal_episode_lengths = []

#         if batch_num % SAVE_INTERVAL == 0:
#             model_save_path = f"policies/teacher_policy_batch_{batch_num}.pth"
#             agent.save_model(model_save_path)
#             wandb.save(model_save_path) # Save checkpoint to W&B

#     env.close()
#     print("\n--- Training Complete ---")
#     final_model_path = "policies/teacher_policy_final.pth"
#     agent.save_model(final_model_path)
#     wandb.save(final_model_path) # Save final model to W&B
#     wandb.finish()

# if __name__ == "__main__":
#     # Create directories if they don't exist
#     import os
#     os.makedirs('policies', exist_ok=True)
#     os.makedirs('models', exist_ok=True) # Although actor_critic.py is written directly,
#                                         # this might be useful if you refactor.
#     train()

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
ENV_NAME = "HumanoidStandup-v4"
LEARNING_RATE = 3e-4 # PPO default in SB3 is often 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2 # This is handled internally by SB3 PPO
PPO_EPOCHS = 10    # This is handled internally by SB3 PPO
MINIBATCH_SIZE = 64 # This is handled internally by SB3 PPO
TIMESTEPS_PER_BATCH = 2048 # This is `n_steps` in SB3 PPO
TOTAL_TIMESTEPS = 5_000_000 # Total environment steps for training
LOG_INTERVAL = 1       # Log every X updates (SB3 logs per training update)
SAVE_INTERVAL = 50000   # Save model every X timesteps (approximate)

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
        if self.locals['dones']:
            for i, done in enumerate(self.locals['dones']):
                if done:
                    episode_reward = self.locals['rewards'][i] # Assuming single env for simplicity here, adjust for vec_env
                    episode_length = self.locals['infos'][i].get('episode', {}).get('l', 0) # Length from info dict
                    
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
        if self.num_timesteps - self.last_log_timesteps >= self.model.n_steps: # Log after each policy update
            self.last_log_timesteps = self.num_timesteps
            
            # Get metrics from SB3's internal logger
            logger_dict = self.logger.get_latest_table_key_values()
            
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
            print(f"Actor Loss: {logger_dict.get('train/policy_loss', 0):.4f}, Critic Loss: {logger_dict.get('train/value_loss', 0):.4f}\n")
            
            self.episode_rewards = []
            self.episode_lengths = []

        # Save model checkpoint
        if self.num_timesteps % SAVE_INTERVAL < self.model.n_steps: # Check if it's time to save
             model_save_path = f"policies/teacher_policy_timesteps_{self.num_timesteps}.zip"
             self.model.save(model_save_path)
             wandb.save(model_save_path)
             print(f"Model saved to {model_save_path}")

        return True # Continue training

# --- Main Training Loop ---
def train():
    # Create the environment. Stable Baselines3 prefers vectorized environments.
    # For rendering, we'll create a single environment and wrap it.
    # Note: SB3's PPO is designed for vectorized envs, so `make_vec_env` is standard.
    # Rendering a vectorized environment can be complex; for a mini-project,
    # often you'd train headless and render later with a loaded policy.
    # For direct rendering during training, a single environment is simpler.
    # If using render_mode='human', it's usually one env, not vectorized.
    env = gym.make(ENV_NAME, render_mode='human')
    # Wrap it in a dummy vec env for SB3 compatibility, if needed for single env rendering
    # Otherwise, for headless training, use: env = make_vec_env(ENV_NAME, n_envs=1)


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
        wandb.init(project="humanoid-drl-teacher", name="ppo_teacher_humanoid_standup_v5_render_sb3", config=model.get_parameters())
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
