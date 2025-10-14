import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

# ----------------------------
# Configuration
# ----------------------------
class Config:
    """Configuration class for hyperparameters and settings."""
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_DIR = "models/student"
    # Path to load the initial student model from
    LOAD_PATH = os.path.join(MODEL_DIR, "student_policy.pth")
    # Path to save the fine-tuned model to
    SAVE_PATH = os.path.join(MODEL_DIR, "student_policy_finetuned.pth")

    # Training
    N_EPOCHS = 50
    STEPS_PER_EPOCH = 5000
    BATCH_SIZE = 256
    LR = 3e-4
    REPLAY_BUFFER_SIZE = 1_000_000

    # Evaluation
    EVAL_EVERY_N_EPOCHS = 5
    EVAL_N_EPISODES = 10

    # Environment
    ENV_NAME = "Humanoid-v5"
    SEED = 42

# ----------------------------
# Replay Buffer
# ----------------------------
class ReplayBuffer:
    """A simple replay buffer for storing and sampling experiences."""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action):
        """Adds an experience to the buffer."""
        self.buffer.append((state, action))

    def sample(self, batch_size):
        """Samples a batch of experiences from the buffer."""
        states, actions = zip(*random.sample(self.buffer, batch_size))
        return np.array(states), np.array(actions)

    def __len__(self):
        return len(self.buffer)

# ----------------------------
# Student Policy
# ----------------------------
class StudentPolicy(nn.Module):
    """A simple MLP policy network."""
    def __init__(self, obs_dim, act_dim, pretrained_state_dict=None):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, act_dim)

        if pretrained_state_dict:
            # Load all layers except the final one to adapt to a new action space.
            sd = {k: v for k, v in pretrained_state_dict.items() if "fc3" not in k}
            self.load_state_dict(sd, strict=False)
            print(f"✅ Pretrained weights loaded. Final layer re-initialized for action_dim={act_dim}")

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

# ----------------------------
# Utility Functions
# ----------------------------
def set_seed(seed, env=None):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if env:
        env.action_space.seed(seed)
        # The reset method will also seed the observation space
        
def load_student_model(obs_dim, act_dim, device):
    """Loads a student model, creating a new one if not found."""
    if os.path.exists(Config.LOAD_PATH):
        print(f"✅ Loading pretrained student from {Config.LOAD_PATH}")
        state_dict = torch.load(Config.LOAD_PATH, map_location=device)
        student = StudentPolicy(obs_dim, act_dim, state_dict).to(device)
    else:
        print(f"⚠️ No pretrained student found at {Config.LOAD_PATH}. Initializing new model.")
        student = StudentPolicy(obs_dim, act_dim).to(device)
    return student

# ----------------------------
# Evaluate Student
# ----------------------------
def evaluate_student(student, env, n_episodes=10):
    """Evaluates the student policy for a number of episodes."""
    student.eval()
    rewards = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_r = 0
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=Config.DEVICE).unsqueeze(0)
            with torch.no_grad():
                action = student(obs_tensor).squeeze(0).cpu().numpy()
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += reward
            done = terminated or truncated
        rewards.append(total_r)
    return np.mean(rewards)

# ----------------------------
# Fine-tune Student
# ----------------------------
def fine_tune_student(student, env):
    """Collects experience and fine-tunes the student policy."""
    optimizer = optim.Adam(student.parameters(), lr=Config.LR)
    replay_buffer = ReplayBuffer(Config.REPLAY_BUFFER_SIZE)
    
    obs, info = env.reset(seed=Config.SEED)
    total_steps = 0
    best_eval_reward = -np.inf
    max_steps = Config.N_EPOCHS * Config.STEPS_PER_EPOCH

    print("🚀 Starting fine-tuning...")
    while total_steps < max_steps:
        # --- Interaction Step ---
        student.eval() # Use eval mode for interaction
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=Config.DEVICE).unsqueeze(0)
        with torch.no_grad():
            action = student(obs_tensor).squeeze(0).cpu().numpy()

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        replay_buffer.push(obs, action)
        obs = next_obs
        total_steps += 1
        
        if done:
            obs, info = env.reset()

        # --- Training Step ---
        if len(replay_buffer) < Config.BATCH_SIZE:
            continue

        student.train()
        obs_batch, act_batch = replay_buffer.sample(Config.BATCH_SIZE)
        
        obs_batch_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=Config.DEVICE)
        act_batch_tensor = torch.tensor(act_batch, dtype=torch.float32, device=Config.DEVICE)

        pred_act = student(obs_batch_tensor)
        loss = F.mse_loss(pred_act, act_batch_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # --- Logging and Evaluation ---
        if total_steps % Config.STEPS_PER_EPOCH == 0:
            epoch = total_steps // Config.STEPS_PER_EPOCH
            print(f"[Epoch {epoch}/{Config.N_EPOCHS}] Steps: {total_steps}, Loss: {loss.item():.6f}")

            if epoch > 0 and epoch % Config.EVAL_EVERY_N_EPOCHS == 0:
                print("\nConducting evaluation...")
                eval_reward = evaluate_student(student, env, n_episodes=Config.EVAL_N_EPISODES)
                print(f"Evaluation at epoch {epoch}: Mean reward = {eval_reward:.2f}")
                
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    os.makedirs(Config.MODEL_DIR, exist_ok=True)
                    torch.save(student.state_dict(), Config.SAVE_PATH)
                    print(f"🎉 New best model! Saved to {Config.SAVE_PATH} with reward {best_eval_reward:.2f}\n")

    print(f"✅ Fine-tuning complete. Best fine-tuned model saved at {Config.SAVE_PATH}")

# ----------------------------
# Main Execution
# ----------------------------
def main():
    """Main function to run the fine-tuning process."""
    # Create environment for training
    env = gym.make(Config.ENV_NAME)
    set_seed(Config.SEED, env)
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    student = load_student_model(obs_dim, act_dim, Config.DEVICE)
    fine_tune_student(student, env)
    
    # Create a separate environment for final evaluation with rendering
    print("\nStarting final evaluation with rendering...")
    eval_env = gym.make(Config.ENV_NAME, render_mode="human")
    set_seed(Config.SEED + 1, eval_env) # Use a different seed for final eval
    
    # Load the best saved model for evaluation
    final_model = load_student_model(obs_dim, act_dim, Config.DEVICE)
    if os.path.exists(Config.SAVE_PATH):
        print(f"Loading best fine-tuned model from {Config.SAVE_PATH} for final evaluation.")
        final_model.load_state_dict(torch.load(Config.SAVE_PATH, map_location=Config.DEVICE))

    evaluate_student(final_model, eval_env, n_episodes=5)

    # Clean up environments
    env.close()
    eval_env.close()
    print("All done.")

if __name__ == "__main__":
    main()
