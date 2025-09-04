"""
SAC for Gymnasium MuJoCo HumanoidStandup-v5 with comprehensive Weights & Biases logging.
- Parallel vectorized training (SubprocVecEnv)
- (Optional) VecNormalize for obs/reward normalization
- Frequent scalar logging to W&B (sync TensorBoard + custom scalar callback)
- Periodic evaluation with best-model checkpointing
- Optional evaluation video logging to W&B

Usage (example):
    python sac_humanoid_standup_v5.py \
        --total-steps 50000000 \
        --n-envs 8 \
        --lr 1e-4 \
        --ent-coef auto \
        --vecnorm

Notes:
- Requires: gymnasium[mujoco], stable-baselines3>=2.3.0, wandb
- If HumanoidStandup-v5 is unavailable in your install, switch to -v4 via --env-name
"""
from __future__ import annotations
import os
import time
import argparse
import numpy as np
import gymnasium as gym
import wandb

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from wandb.integration.sb3 import WandbCallback

# ----------------------------
# Extra scalar logging callback
# ----------------------------
class ScalarLogCallback(BaseCallback):
    """Push SB3 logger scalars to W&B at a fixed step interval."""
    def __init__(self, log_interval_steps: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval_steps
        self._last = 0
        self._t0 = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last >= self.log_interval:
            self._last = self.num_timesteps
            d = self.logger.name_to_value
            keys = [
                # rollout
                "rollout/ep_rew_mean", "rollout/ep_len_mean",
                # train losses
                "train/actor_loss", "train/critic_loss", "train/alpha_loss", "train/ent_coef",
                # timing
                "time/episodes", "time/fps",
            ]
            payload = {k: d[k] for k in keys if k in d}
            payload["global_timesteps"] = self.num_timesteps
            payload["time_elapsed_s"] = time.time() - self._t0
            if payload:
                wandb.log(payload, step=self.num_timesteps)
        return True

# ----------------------------
# Simple helper to make an eval env with optional video recording
# ----------------------------

def make_eval_env(env_name: str, seed: int, record_dir: str | None = None, vecnorm_source: VecNormalize | None = None):
    def _make():
        e = gym.make(env_name, render_mode="rgb_array")
        e = Monitor(e)
        if record_dir is not None:
            # RecordVideo wrapper records when reset(..., options={"video_folder"...}) or every episode via trigger
            e = gym.wrappers.RecordVideo(e, video_folder=record_dir, episode_trigger=lambda ep: True)
        return e
    eval_env = _make()
    if vecnorm_source is not None:
        # Create a VecNormalize with same statistics for evaluation
        eval_vec = VecNormalize(venv=gym.vector.SyncVectorEnv([lambda: eval_env]),
                                training=False, norm_obs=True, norm_reward=False,
                                clip_obs=vecnorm_source.clip_obs)
        # load running stats
        eval_vec.obs_rms = vecnorm_source.obs_rms
        return eval_vec
    return eval_env

# ----------------------------
# Main training
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--project", type=str, default="HumanoidStandup-SAC")
    parser.add_argument("--project", type=str, default="humanoid-drl-teacher")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--env-name", type=str, default="HumanoidStandup-v5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--total-steps", type=int, default=50_000_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=250_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--ent-coef", type=str, default="auto")
    parser.add_argument("--learning-starts", type=int, default=40_000)
    parser.add_argument("--gradient-steps", type=int, default=8)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--vecnorm", action="store_true", help="Use VecNormalize for obs/reward")
    parser.add_argument("--clip-obs", type=float, default=10.0)
    parser.add_argument("--clip-rew", type=float, default=10.0)
    parser.add_argument("--log-interval-steps", type=int, default=5000)
    parser.add_argument("--save-video", action="store_true", help="Log evaluation videos to W&B")
    args = parser.parse_args()

    set_random_seed(args.seed)

    # ---- W&B init ----
    run_name = args.run_name or f"SAC_{args.env_name}_{int(time.time())}"
    run = wandb.init(
        project=args.project,
        name=run_name,
        config=vars(args),
        sync_tensorboard=True,   # stream TB scalars to W&B
        monitor_gym=True,
        save_code=True,
    )

    # ---- Training VecEnv ----
    env = make_vec_env(
        args.env_name,
        n_envs=args.n_envs,
        vec_env_cls=SubprocVecEnv,
        seed=args.seed,
    )

    if args.vecnorm:
        env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True,
                           clip_obs=args.clip_obs, clip_reward=args.clip_rew)

    # Derive target entropy: good default is -|A|
    action_dim = int(np.prod(env.action_space.shape))
    target_entropy = -float(action_dim)

    save_root = os.path.join("models", run.id)
    os.makedirs(save_root, exist_ok=True)

    # ---- Callbacks ----
    wandb_cb = WandbCallback(model_save_path=save_root, verbose=0)

    # Evaluation env (1 env). If using VecNormalize, we will share stats after learning starts.
    eval_env = make_vec_env(args.env_name, n_envs=1, seed=args.seed + 123)
    if args.vecnorm:
        eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False,
                                clip_obs=args.clip_obs)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=save_root,
        eval_freq=max(10_000 // max(args.n_envs, 1), 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    scalar_cb = ScalarLogCallback(log_interval_steps=args.log_interval_steps)

    # ---- SAC model ----
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        ent_coef=args.ent_coef,
        learning_starts=args.learning_starts,
        train_freq=(args.train_freq, "step"),
        gradient_steps=args.gradient_steps,
        target_entropy=target_entropy,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
        seed=args.seed,
        device="auto",
    )

    # Helper to persist VecNormalize stats alongside checkpoints
    def save_vecnorm_stats(path: str):
        if isinstance(env, VecNormalize):
            env.save(os.path.join(path, "vecnormalize.pkl"))

    # ---- Train ----
    print(f"🚀 Training SAC on {args.env_name} for {args.total_steps} steps with {args.n_envs} envs…")
    model.learn(
        total_timesteps=args.total_steps,
        callback=[wandb_cb, eval_cb, scalar_cb],
        log_interval=1,
        progress_bar=True,
    )

    # ---- Save final model & VecNormalize ----
    final_path = os.path.join(save_root, "teacher_policy_final_sac.zip")
    model.save(final_path)
    save_vecnorm_stats(save_root)
    try:
        wandb.save(final_path)
        if os.path.exists(os.path.join(save_root, "vecnormalize.pkl")):
            wandb.save(os.path.join(save_root, "vecnormalize.pkl"))
    except Exception:
        pass

    # ---- Optional: record & log evaluation video ----
    if args.save_video:
        video_dir = os.path.join(save_root, "videos")
        os.makedirs(video_dir, exist_ok=True)
        # Create a single non-vector env with video recorder
        raw_eval = gym.make(args.env_name, render_mode="rgb_array")
        raw_eval = Monitor(raw_eval)
        raw_eval = gym.wrappers.RecordVideo(raw_eval, video_folder=video_dir, episode_trigger=lambda ep: True)
        obs, _ = raw_eval.reset()
        for ep in range(3):
            done = False
            trunc = False
            while not (done or trunc):
                # If VecNormalize used, normalize obs with stored stats for fair eval
                if isinstance(env, VecNormalize):
                    # use saved running mean/std from training
                    obs_in = env.normalize_obs(obs)
                else:
                    obs_in = obs
                action, _ = model.predict(obs_in, deterministic=True)
                obs, _, done, trunc, _ = raw_eval.step(action)
            raw_eval.reset()
        raw_eval.close()
        # Log the latest video to W&B
        try:
            # wandb.Video expects a path to a video file; RecordVideo saves mp4 files
            for fname in sorted(os.listdir(video_dir)):
                if fname.endswith(".mp4"):
                    wandb.log({"eval_video": wandb.Video(os.path.join(video_dir, fname), fps=30, format="mp4")})
        except Exception:
            pass

    run.finish()
    print("✅ Training complete.")


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    main()
