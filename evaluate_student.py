import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

# ----------------------------
# Configuration
# ----------------------------
# --- IMPORTANT: Path to the model you want to evaluate ---
# This should be the model saved by your fine-tuning script.
MODEL_PATH = "models/student/student_policy_finetuned.pth"
ENV_NAME = "Humanoid-v5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_EPISODES = 5 # Number of episodes to run for visualization

# ----------------------------
# Student Policy Definition
# ----------------------------
# This class must match the architecture of the saved model EXACTLY.
class StudentPolicy(nn.Module):
    """A simple MLP policy network."""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, act_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

# ----------------------------
# Main Evaluation Function
# ----------------------------
def main():
    """Loads a trained policy and evaluates it in the environment."""
    print("🚀 Starting policy evaluation...")
    print(f"Environment: {ENV_NAME}")
    print(f"Device: {DEVICE}")

    # --- 1. Check if the model file exists ---
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found at '{MODEL_PATH}'")
        print("Please make sure you have run the fine-tuning script first and the path is correct.")
        return

    # --- 2. Create the environment with rendering ---
    # The 'render_mode="human"' argument is what creates the visualization window.
    env = gym.make(ENV_NAME, render_mode="human",max_episode_steps=5000)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # --- 3. Load the trained policy ---
    policy = StudentPolicy(obs_dim, act_dim).to(DEVICE)
    policy.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    policy.eval() # Set the model to evaluation mode (important!)
    print(f"✅ Policy loaded successfully from '{MODEL_PATH}'")

    # --- 4. Run the evaluation loop ---
    for episode in range(N_EPISODES):
        obs, info = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        print(f"\n--- Starting Episode {episode + 1}/{N_EPISODES} ---")

        while not done:
            # Convert observation to a PyTorch tensor
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            # Get action from the policy
            with torch.no_grad(): # Disable gradient calculation for inference
                action = policy(obs_tensor).cpu().numpy().squeeze()

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            done = terminated or truncated
            
            # Optional: a small delay to make rendering smoother
            time.sleep(0.01)

        print(f"Episode finished after {step_count} steps.")
        print(f"Total Reward: {total_reward:.2f}")

    # --- 5. Clean up ---
    env.close()
    print("\n✅ Evaluation complete.")


if __name__ == "__main__":
    main()
