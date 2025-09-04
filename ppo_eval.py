import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

# Load trained model
model_path = "models/ppo1/best_model.zip"
vecnorm_path = "models/ppo1/vecnormalize.pkl"

# Load environment (must match training)
eval_env = gym.make("HumanoidStandup-v5")

# Load VecNormalize stats
eval_env = VecNormalize.load(vecnorm_path, eval_env)
eval_env.training = False      # important: don’t update stats
eval_env.norm_reward = False   # important: keep raw rewards for eval

# Load model
model = PPO.load(model_path, env=eval_env, device="cuda")

# Run evaluation
obs = eval_env.reset()
done = False
ep_reward = 0
while not done:
    action, _ = model.predict(obs, deterministic=True)  # <-- deterministic!
    obs, reward, done, info = eval_env.step(action)
    ep_reward += reward

print(f"Episode Reward: {ep_reward}")
