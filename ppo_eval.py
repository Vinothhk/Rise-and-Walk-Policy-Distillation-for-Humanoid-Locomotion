# evaluate_ppo.py
import os, time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecEnv

ENV_NAME   = "HumanoidStandup-v5"
RUN_DIR    = "models/ppo1"                # folder that holds best_model.zip + vecnormalize.pkl
MODEL_FILE = "best_model.zip"             # or "teacher_policy_final_ppo.zip"
VEC_FILE   = "vecnormalize.pkl"           # must exist for 1:1 performance
RENDER     = True                         # set False for headless eval
EPISODES   = 5

def make_render_env(render: bool) -> VecEnv:
    def _make():
        return gym.make(ENV_NAME, render_mode="human" if render else None)
    return DummyVecEnv([_make])

def main():
    model_path = os.path.join(RUN_DIR, MODEL_FILE)
    vec_path   = os.path.join(RUN_DIR, VEC_FILE)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(vec_path):
        raise FileNotFoundError(f"VecNormalize file not found: {vec_path}")

    # Base env
    base_env = make_render_env(RENDER)

    # Load VecNormalize stats onto the base env
    eval_env = VecNormalize.load(vec_path, base_env)
    eval_env.training = False     # do not update stats
    eval_env.norm_reward = False  # report raw rewards (comparable to real env reward)

    # Load model with the normalized eval env
    model = PPO.load(model_path, env=eval_env, device="auto")

    print("👀 Running trained agent with synchronized VecNormalize...")
    for ep in range(EPISODES):
        obs = eval_env.reset()
        done = False
        ep_reward = 0.0

        # VecEnv uses arrays; unwrap flags carefully
        terminated, truncated = False, False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            terminated, truncated = done, False
            ep_reward += float(reward)

            if RENDER:
                time.sleep(1/120)


        print(f"Episode {ep+1}: Total Reward = {ep_reward:.2f}")

    eval_env.close()

if __name__ == "__main__":
    main()
