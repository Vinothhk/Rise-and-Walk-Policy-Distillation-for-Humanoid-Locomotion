# teacher_train_sac.py
import os, time
import numpy as np
import gymnasium as gym
import wandb

from stable_baselines3 import SAC
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
    "total_timesteps": 50_000_000,
    "learning_rate": 3e-4,
    "buffer_size": 1_000_000,
    "batch_size": 512,
    "gamma": 0.99,
    "tau": 0.005,
    "ent_coef": "auto_0.1",
    "learning_starts": 150_000,
    "train_freq_steps": 1,
    "gradient_steps": 1,
    "policy_kwargs": dict(net_arch=dict(pi=[512, 512, 256], qf=[512, 512, 256])),
    "log_interval_steps": 2_000,
    "use_vecnormalize": True,
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
            # Pick common SAC keys if present:
            keys = [
                "rollout/ep_rew_mean", "rollout/ep_len_mean",
                "train/actor_loss", "train/critic_loss",
                "train/alpha_loss", "train/ent_coef",
                "time/episodes", "time/fps"
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
    RUN_NAME = f"SAC_{config['env_name']}_{int(time.time())}"

    # Init W&B (sync TensorBoard → denser W&B charts)
    run = wandb.init(
        project=PROJECT,
        name=RUN_NAME,
        config=config,
        sync_tensorboard=True,   # <- ensures SB3 TB scalars stream into W&B frequently
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
            clip_reward=config["clip_reward"]
        )

    # Derive target entropy (good default: -|A|)
    # Note: for VecEnv, action_space is the same as single env's
    action_dim = int(np.prod(env.action_space.shape))
    target_entropy = -float(action_dim)

    # Save location
    save_path = f"models/{run.id}"
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
            clip_obs=config["clip_obs"]
        )
        # sync stats later after model.learn starts writing them

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        eval_freq=max(10_000 // config["n_envs"], 1),  # your requested scaling
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    # Extra scalar logging
    scalar_callback = ScalarLogCallback(log_interval_steps=config["log_interval_steps"])

    # Build SAC
    model = SAC(
        policy=config["policy_type"],
        env=env,
        learning_rate=config["learning_rate"],
        buffer_size=config["buffer_size"],
        batch_size=config["batch_size"],
        gamma=config["gamma"],
        tau=config["tau"],
        ent_coef=config["ent_coef"],
        learning_starts=config["learning_starts"],
        train_freq=(config["train_freq_steps"], "step"),
        gradient_steps=config["gradient_steps"],
        target_entropy=target_entropy,                 # <-- explicit target entropy
        policy_kwargs=config["policy_kwargs"],
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
        seed=config["seed"],
        device="auto"
    )

    # If using VecNormalize, make sure we periodically save stats
    # Simple approach: save at the end; you can also save on best eval.
    def save_vecnormalize_stats(path):
        if isinstance(env, VecNormalize):
            env.save(os.path.join(path, "vecnormalize.pkl"))

    print(f"🚀 Starting SAC for {config['total_timesteps']} steps on {config['n_envs']} envs...")
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=[wandb_callback, eval_callback, scalar_callback],
        log_interval=1,          # SB3 logger ticks every training iteration
        progress_bar=True
    )

    # Save final artifacts
    final_model = os.path.join(save_path, "teacher_policy_final_sac.zip")
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
