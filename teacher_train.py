# teacher_train_ppo_tuned.py
"""
Improved PPO trainer for HumanoidStandup-v5.

- Larger networks, larger batch sizes, linear LR schedule
- Autosave VecNormalize stats and best_model
- Eval + checkpoint callbacks
- Optional quick local sweep over a few hyperparam combos for fast checks

Run:
    python teacher_train_ppo_tuned.py --run-name test_run

Quick local sweep (short runs for comparison):
    python teacher_train_ppo_tuned.py --local-sweep --sweep-steps 2000000

Notes:
 - Run from a terminal (not an interactive notebook) due to SubprocVecEnv.
 - Requires stable-baselines3, gymnasium[mujoco], wandb.
"""
from __future__ import annotations
import os, time, argparse
from typing import Dict, Any, List, Tuple
import numpy as np
import wandb
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize, VecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from wandb.integration.sb3 import WandbCallback

# ----------------------------
# Callbacks
# ----------------------------
class AutoSaveVecNormCallback(BaseCallback):
    """Periodically save VecNormalize stats to disk and at training end."""
    def __init__(self, save_dir: str, every_steps: int = 100_000, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.every_steps = every_steps
        self._last = 0
        os.makedirs(self.save_dir, exist_ok=True)

    def _save(self):
        env = self.model.get_env()
        if isinstance(env, VecNormalize):
            path = os.path.join(self.save_dir, "vecnormalize.pkl")
            env.save(path)
            if self.verbose:
                print(f"[AutoSaveVecNorm] saved to {path}")

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last >= self.every_steps:
            self._last = self.num_timesteps
            self._save()
        return True

    def _on_training_end(self) -> None:
        self._save()

class ScalarLogCallback(BaseCallback):
    """Push SB3 scalar logger values to W&B at an interval."""
    def __init__(self, log_interval_steps: int = 2000, verbose: int = 0):
        super().__init__(verbose)
        self.interval = log_interval_steps
        self._last = 0
        self._t0 = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last >= self.interval:
            self._last = self.num_timesteps
            d = self.logger.name_to_value
            keys = [
                "rollout/ep_rew_mean", "rollout/ep_len_mean",
                "train/policy_loss", "train/value_loss",
                "train/entropy_loss", "train/approx_kl",
                "train/clip_fraction", "train/explained_variance",
                "time/iterations", "time/fps"
            ]
            payload = {k: d[k] for k in keys if k in d}
            payload["global_timesteps"] = self.num_timesteps
            payload["time_elapsed_s"] = time.time() - self._t0
            if payload:
                wandb.log(payload, step=self.num_timesteps)
        return True

# ----------------------------
# Helpers (env builders)
# ----------------------------
def build_train_env(env_name: str, n_envs: int, seed: int, use_vecnorm: bool,
                    clip_obs: float, clip_reward: float) -> VecEnv:
    env = make_vec_env(env_name, n_envs=n_envs, vec_env_cls=SubprocVecEnv, seed=seed)
    if use_vecnorm:
        env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True,
                           clip_obs=clip_obs, clip_reward=clip_reward)
    return env

def build_eval_env(env_name: str, seed: int, use_vecnorm: bool, clip_obs: float) -> VecEnv:
    env = make_vec_env(env_name, n_envs=1, vec_env_cls=DummyVecEnv, seed=seed+123)
    if use_vecnorm:
        env = VecNormalize(env, training=False, norm_obs=True, norm_reward=False, clip_obs=clip_obs)
    return env

# ----------------------------
# LR schedule helper
# ----------------------------
def linear_schedule(initial_value: float):
    def lr_fn(progress_remaining: float):
        return progress_remaining * initial_value
    return lr_fn

# ----------------------------
# Single-run trainer
# ----------------------------
def run_single(cfg: Dict[str, Any], run_name: str):
    # init W&B
    run = wandb.init(project=cfg["project"], name=run_name, config=cfg, sync_tensorboard=True, reinit=True)

    save_path = cfg.get("save_path") or f"models/{run.id}"
    os.makedirs(save_path, exist_ok=True)

    print("Building train & eval envs...")
    train_env = build_train_env(cfg["env_name"], cfg["n_envs"], cfg["seed"], cfg["use_vecnormalize"], cfg["clip_obs"], cfg["clip_reward"])
    eval_env = build_eval_env(cfg["env_name"], cfg["seed"], cfg["use_vecnormalize"], cfg["clip_obs"])

    # callbacks
    wandb_cb = WandbCallback(model_save_path=save_path, verbose=0)
    scalar_cb = ScalarLogCallback(log_interval_steps=cfg["log_interval_steps"])
    autosave_cb = AutoSaveVecNormCallback(save_dir=save_path, every_steps=cfg["vecnorm_save_every"], verbose=1)
    checkpoint_cb = CheckpointCallback(save_freq=cfg["checkpoint_every"], save_path=save_path, name_prefix="ppo_cp")

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=save_path,
        eval_freq=cfg["eval_freq"],
        n_eval_episodes=cfg["n_eval_episodes"],
        deterministic=True,
        render=False,
    )

    # policy kwargs
    policy_kwargs = dict(net_arch=dict(pi=cfg["pi_arch"], vf=cfg["vf_arch"]))

    # LR schedule (callable supported by SB3)
    lr = linear_schedule(cfg["learning_rate"]) if cfg["use_lr_schedule"] else cfg["learning_rate"]

    # build model
    print("Creating PPO model...")
    model = PPO(
        policy=cfg["policy_type"],
        env=train_env,
        learning_rate=lr,
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_range"],
        ent_coef=cfg["ent_coef"],
        vf_coef=cfg["vf_coef"],
        max_grad_norm=cfg["max_grad_norm"],
        target_kl=cfg["target_kl"],
        use_sde=cfg["use_sde"],
        sde_sample_freq=cfg["sde_sample_freq"],
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
        seed=cfg["seed"],
        device="auto",
    )

    callbacks = [wandb_cb, eval_cb, autosave_cb, scalar_cb, checkpoint_cb]
    print("Starting training...")
    model.learn(total_timesteps=cfg["total_timesteps"], callback=callbacks, log_interval=1, progress_bar=True)

    # final save
    final_model = os.path.join(save_path, "teacher_policy_final_ppo.zip")
    model.save(final_model)
    if isinstance(train_env, VecNormalize):
        train_env.save(os.path.join(save_path, "vecnormalize.pkl"))

    wandb.save(final_model)
    if isinstance(train_env, VecNormalize):
        wandb.save(os.path.join(save_path, "vecnormalize.pkl"))
    run.finish()
    print("Training finished. Artifacts saved to:", save_path)

# ----------------------------
# Quick local sweep runner (sequential)
# ----------------------------
def run_local_sweep(base_cfg: Dict[str, Any], variants: List[Dict[str, Any]], steps_per_trial: int):
    for i, v in enumerate(variants):
        cfg = base_cfg.copy()
        cfg.update(v)
        cfg["total_timesteps"] = steps_per_trial
        run_name = f"local_sweep_{i}_{int(time.time())}"
        print(f"\n=== Local sweep trial {i} - run name: {run_name} - params: {v} ===\n")
        run_single(cfg, run_name)

# ----------------------------
# Entrypoint & default config
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--project", type=str, default="HumanoidStandup-Optimal")
    parser.add_argument("--local-sweep", action="store_true", help="Run small local hyperparam comparison (sequential short runs)")
    parser.add_argument("--sweep-steps", type=int, default=2_000_000, help="steps for each local sweep trial")
    parser.add_argument("--save-path", type=str, default=None)
    args = parser.parse_args()

    # Base configuration tuned to push beyond ~2.5e5
    base_cfg: Dict[str, Any] = dict(
        project=args.project,
        save_path=args.save_path,
        env_name="HumanoidStandup-v5",
        policy_type="MlpPolicy",
        n_envs=16,                   # more parallel envs
        total_timesteps=35_000_000,  # increase training budget
        learning_rate=3e-4,
        use_lr_schedule=True,
        n_steps=4096,                # large rollout buffer
        batch_size=4096,
        n_epochs=20,
        gamma=0.997,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,              # slightly reduce entropy bonus
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.02,
        use_sde=True,
        sde_sample_freq=4,
        pi_arch=[1024, 1024, 512],
        vf_arch=[1024, 1024, 512],
        use_vecnormalize=True,
        clip_obs=10.0,
        clip_reward=50.0,           # allow larger reward signal before clipping
        log_interval_steps=2000,
        vecnorm_save_every=100_000,
        eval_freq=20_000,
        n_eval_episodes=8,
        checkpoint_every=500_000,
        seed=0,
    )

    # If local sweep requested, try a few promising combos quickly
    if args.local_sweep:
        # variants to try: combos of lr and ent_coef (and optionally n_steps)
        variants = [
            {"learning_rate": 3e-4, "ent_coef": 0.005},
            {"learning_rate": 2e-4, "ent_coef": 0.005},
            {"learning_rate": 1.5e-4, "ent_coef": 0.01},
            {"learning_rate": 4e-4, "ent_coef": 0.0025},
        ]
        run_local_sweep(base_cfg, variants, steps_per_trial=args.sweep_steps)
    else:
        run_name = args.run_name or f"ppo_tuned_{int(time.time())}"
        run_single(base_cfg, run_name)

if __name__ == "__main__":
    main()
