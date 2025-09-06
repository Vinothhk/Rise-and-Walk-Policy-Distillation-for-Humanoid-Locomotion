# teacher_train_ppo.py
import os, time
import numpy as np
import gymnasium as gym
import wandb
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize, VecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from wandb.integration.sb3 import WandbCallback  # official SB3↔W&B


# ----------------------------
# Config (tweak here)
# ----------------------------
config = {
    "policy_type": "MlpPolicy",
    "env_name": "HumanoidStandup-v5",
    "n_envs": 8,

    # PPO is on-policy — use large batches
    "total_timesteps": 25_000_000,
    "learning_rate": 3e-4,
    "n_steps": 4096,                # per-env steps → batch = n_steps * n_envs
    "batch_size": 2048,
    "n_epochs": 10,

    "gamma": 0.995,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": 0.03,

    "use_sde": True,
    "sde_sample_freq": 4,

    "policy_kwargs": dict(net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256])),

    "use_vecnormalize": True,
    "clip_obs": 10.0,
    "clip_reward": 10.0,

    "log_interval_steps": 2_000,    # extra scalar push cadence to W&B
    "vecnorm_save_every": 100_000,  # save vecnorm periodically
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
# Periodically save VecNormalize stats (and on training end)
# ----------------------------
class AutoSaveVecNormCallback(BaseCallback):
    def __init__(self, save_dir: str, every_steps: int, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.every_steps = every_steps
        self.last = 0
        os.makedirs(self.save_dir, exist_ok=True)

    def _save_vecnorm(self):
        env = self.model.get_env()
        if isinstance(env, VecNormalize):
            path = os.path.join(self.save_dir, "vecnormalize.pkl")
            env.save(path)
            if self.verbose:
                print(f"[AutoSaveVecNorm] Saved VecNormalize to {path}")

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last >= self.every_steps:
            self.last = self.num_timesteps
            self._save_vecnorm()
        return True

    def _on_training_end(self) -> None:
        self._save_vecnorm()

# ----------------------------
# Eval callback that syncs VecNormalize stats from train env -> eval env
# ----------------------------
class SyncedEvalCallback(EvalCallback):
    def __init__(self, eval_env: VecEnv, train_env: VecEnv, save_path: str, **kwargs):
        super().__init__(eval_env, best_model_save_path=save_path, **kwargs)
        self.train_env = train_env
        self.save_path = save_path

    def _sync_vecnorm(self):
        if isinstance(self.train_env, VecNormalize) and isinstance(self.eval_env, VecNormalize):
            # copy running mean/var (obs only; leave reward raw for reporting)
            self.eval_env.obs_rms = self.train_env.obs_rms

    def _on_step(self) -> bool:
        self._sync_vecnorm()
        return super()._on_step()

# ----------------------------
# Helpers
# ----------------------------
def build_train_env(cfg) -> VecEnv:
    env = make_vec_env(
        cfg["env_name"],
        n_envs=cfg["n_envs"],
        vec_env_cls=SubprocVecEnv,
        seed=cfg["seed"]
    )
    if cfg["use_vecnormalize"]:
        env = VecNormalize(
            env,
            training=True,
            norm_obs=True,
            norm_reward=True,
            clip_obs=cfg["clip_obs"],
            clip_reward=cfg["clip_reward"],
        )
    return env

def build_eval_env(cfg) -> VecEnv:
    # use single-process env for eval; no render for speed
    env = make_vec_env(cfg["env_name"], n_envs=1, vec_env_cls=DummyVecEnv, seed=cfg["seed"] + 123)
    if cfg["use_vecnormalize"]:
        env = VecNormalize(
            env,
            training=False,         # do not update stats
            norm_obs=True,
            norm_reward=False,      # report raw rewards
            clip_obs=cfg["clip_obs"]
        )
    return env

# ----------------------------
# Main
# ----------------------------
def train():
    PROJECT = "HumanoidStandup-Optimal"
    RUN_NAME = f"PPO_{config['env_name']}_{int(time.time())}"

    run = wandb.init(
        project=PROJECT,
        name=RUN_NAME,
        config=config,
        sync_tensorboard=True,   # stream SB3 TB scalars into W&B
        monitor_gym=True,
        save_code=True,
    )

    # Envs
    env = build_train_env(config)
    eval_env = build_eval_env(config)

    # Save location
    save_path = "models/ppo1"
    os.makedirs(save_path, exist_ok=True)

    # Callbacks
    wandb_callback = WandbCallback(
        model_save_path=save_path,
        verbose=0,
    )

    scalar_callback = ScalarLogCallback(log_interval_steps=config["log_interval_steps"])

    autosave_vecnorm = AutoSaveVecNormCallback(
        save_dir=save_path,
        every_steps=config["vecnorm_save_every"],
        verbose=1
    )

    eval_callback = SyncedEvalCallback(
        eval_env=eval_env,
        train_env=env,
        save_path=save_path,
        eval_freq=max(5_000 // config["n_envs"], 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1
    )

    # Model
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

    print(f"🚀 Starting PPO for {config['total_timesteps']} steps on {config['n_envs']} envs...")
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=[wandb_callback, eval_callback, autosave_vecnorm, scalar_callback],
        log_interval=1,
        progress_bar=True
    )

    # Save final artifacts
    final_model = os.path.join(save_path, "teacher_policy_final_ppo.zip")
    model.save(final_model)

    # Final vecnorm snapshot
    if isinstance(env, VecNormalize):
        env.save(os.path.join(save_path, "vecnormalize.pkl"))

    wandb.save(final_model)
    if isinstance(env, VecNormalize):
        wandb.save(os.path.join(save_path, "vecnormalize.pkl"))

    run.finish()
    print("✅ Training finished.")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train()
