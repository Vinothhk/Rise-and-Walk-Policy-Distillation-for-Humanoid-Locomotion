# File: evaluate.py

import gymnasium as gym
from stable_baselines3 import SAC,PPO
import time
import os

# --- IMPORTANT: UPDATE THIS PATH ---
# Path to the saved model, found in 'models/{run_id}/best_model.zip'
MODEL_PATH = "models/ppo1/best_model.zip" 
ENV_NAME = "HumanoidStandup-v5"
# ------------------------------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}.")

env = gym.make(ENV_NAME, render_mode="human")
model = PPO.load(MODEL_PATH, env=env)

print("👀 Running trained agent...")
for ep in range(5):
    obs, info = env.reset()
    terminated, truncated = False, False
    episode_reward = 0
    while not terminated and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        time.sleep(1/60)
    print(f"Episode {ep + 1}: Total Reward = {episode_reward:.2f}")

env.close()