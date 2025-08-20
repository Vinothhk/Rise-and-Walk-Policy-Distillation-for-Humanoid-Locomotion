import gymnasium as gym

import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

env = gym.make('HumanoidStandup-v5', impact_cost_weight=0.5e-6,render_mode='human')

for _ in range(10):
    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"Reward: {reward}, Info: {info}")
        env.render()
env.close()