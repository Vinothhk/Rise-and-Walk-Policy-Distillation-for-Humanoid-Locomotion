# teacher_train_ppo.py
import os, time
import numpy as np
import gymnasium as gym
import wandb

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from wandb.integration.sb3 import WandbCallback  # official SB3↔W&B

# ----------------------------
# Config (tweak here)
# ----------------------------
config = {
    "policy_type": "MlpPolicy",
    "env_name": "HumanoidStandup-v5",
    "n_envs": 8,
    # PPO is on-policy → needs a lot of samples per update. This setup makes big batches.
    "total_timesteps": 100_000_000,  # PPO typically needs more than SAC
    "learning_rate": 3e-4,
    "n_steps": 4096,                 # per-env steps → total batch = n_steps * n_envs (here: 32,768)
    "batch_size": 2048,              # minibatch size for SGD
    "n_epochs": 10,                  # SGD passes
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,                # a bit of entropy for exploration
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": 0.03,               # early stop if PPO update drifts too much
    "use_sde": True,                 # State Dependent Exploration helps on MuJoCo
    "sde_sample_freq": 4,
    "policy_kwargs": dict(net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256])),
    "log_interval_steps": 2_000,     # extra scalar push cadence to W&B
    "use_vecnormalize": True,        # normalize obs & rewards — often crucial on HumanoidStandup
    "clip_obs": 10.0,
    "clip_reward": 10.0,
    "seed": 0,
}

# ----------------------------
# Extra scalar logging callback (push SB3 scalars to W&B more often)
# ----------------------------
class ScalarLogCallback(BaseCallback):
    def __init__(self, log_interval_steps: int, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval_steps
        self.last = 0
        self.t0 = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last >= self.log_interval:
            self.last = self.num_timesteps
            d = self.logger.name_to_value  # SB3 scalar dict
            # Common PPO scalars:
            keys = [
                "rollout/ep_rew_mean", "rollout/ep_len_mean",
                "train/policy_loss", "train/value_loss",
                "train/entropy_loss", "train/approx_kl",
                "train/clip_fraction", "train/explained_variance",
                "time/iterations", "time/fps",
            ]
            payload = {k: d[k] for k in keys if k in d}
            payload["global_timesteps"] = self.num_timesteps
            payload["time_elapsed_s"] = time.time() - self.t0
            if payload:
                wandb.log(payload, step=self.num_timesteps)
        return True

# ----------------------------
# Main
# ----------------------------
def train():
    PROJECT = "HumanoidStandup-Optimal-SAC"
    RUN_NAME = f"PPO_{config['env_name']}_{int(time.time())}"

    # Init W&B (sync TensorBoard → denser W&B charts)
    run = wandb.init(
        project=PROJECT,
        name=RUN_NAME,
        config=config,
        sync_tensorboard=True,   # stream SB3 TB scalars into W&B
        monitor_gym=True,
        save_code=True,
    )

    # Vectorized training env
    env = make_vec_env(
        config["env_name"],
        n_envs=config["n_envs"],
        vec_env_cls=SubprocVecEnv,
        seed=config["seed"]
    )

    # Optional but recommended: normalize obs and rewards
    if config["use_vecnormalize"]:
        env = VecNormalize(
            env,
            training=True,
            norm_obs=True,
            norm_reward=True,
            clip_obs=config["clip_obs"],
            clip_reward=config["clip_reward"],
        )

    # Save location
    save_path = f"models/ppo1"
    os.makedirs(save_path, exist_ok=True)

    # W&B callback (your requested layout)
    wandb_callback = WandbCallback(
        model_save_path=save_path,
        verbose=0,
    )

    # Eval env (separate; no render for speed)
    eval_env = make_vec_env(config["env_name"], n_envs=1, seed=config["seed"] + 123)
    if config["use_vecnormalize"]:
        # IMPORTANT: share normalization stats with eval for meaningful scores
        eval_env = VecNormalize(
            eval_env,
            training=False,
            norm_obs=True,
            norm_reward=False,  # keep raw reward for reporting
            clip_obs=config["clip_obs"],
        )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        eval_freq=max(5_000 // config["n_envs"], 1),  # evaluate fairly often
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    # Extra scalar logging
    scalar_callback = ScalarLogCallback(log_interval_steps=config["log_interval_steps"])

    # Build PPO
    model = PPO(
        policy=config["policy_type"],
        env=env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"],
        target_kl=config["target_kl"],
        use_sde=config["use_sde"],
        sde_sample_freq=config["sde_sample_freq"],
        policy_kwargs=config["policy_kwargs"],
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
        seed=config["seed"],
        device="auto",
    )

    # If using VecNormalize, make sure we periodically save stats
    def save_vecnormalize_stats(path):
        if isinstance(env, VecNormalize):
            env.save(os.path.join(path, "vecnormalize.pkl"))

    print(f"🚀 Starting PPO for {config['total_timesteps']} steps on {config['n_envs']} envs...")
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=[wandb_callback, eval_callback, scalar_callback],
        log_interval=1,          # SB3 logger ticks every update
        progress_bar=True
    )

    # Save final artifacts
    final_model = os.path.join(save_path, "teacher_policy_final_ppo.zip")
    model.save(final_model)
    save_vecnormalize_stats(save_path)
    wandb.save(final_model)
    if isinstance(env, VecNormalize):
        wandb.save(os.path.join(save_path, "vecnormalize.pkl"))

    run.finish()
    print("✅ Training finished.")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train()
