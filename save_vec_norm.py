# resume_and_save_vecnorm.py
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

# ----------------------------
# Config
# ----------------------------
RUN_ID = "ppo1"   # e.g. "models/abc123"
ENV_NAME = "HumanoidStandup-v5"
N_ENVS = 4
SEED = 0

MODEL_PATH = os.path.join("models", RUN_ID, "best_model.zip")
VECNORM_PATH = os.path.join("models", RUN_ID, "vecnormalize.pkl")

def main():
    # Build training env
    env = make_vec_env(ENV_NAME, n_envs=N_ENVS, seed=SEED, vec_env_cls=SubprocVecEnv)
    env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True)

    # Load PPO into this env
    model = PPO.load(MODEL_PATH, env=env, device="auto")

    # Resume training briefly (10k steps is enough)
    print("🔄 Resuming briefly to capture VecNormalize stats...")
    model.learn(total_timesteps=10_000, reset_num_timesteps=False)

    # Save VecNormalize stats
    env.save(VECNORM_PATH)
    print(f"✅ VecNormalize stats saved at {VECNORM_PATH}")

if __name__ == "__main__":
    main()
