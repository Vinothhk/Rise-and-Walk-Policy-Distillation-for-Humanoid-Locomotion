# student_train_distill_fixed.py
import os, time
import numpy as np
import gymnasium as gym
import wandb
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env


# ----------------------------
# Config
# ----------------------------
config: Dict[str, Any] = dict(
    project="HumanoidStandup-Distill",
    env_name="HumanoidStandup-v5",
    teacher_path="models/ppo1/best_model.zip",
    vecnorm_path="models/ppo1/vecnormalize.pkl",

    n_epochs=50,
    batch_size=256,
    lr=3e-4,
    device="cuda" if torch.cuda.is_available() else "cpu",
    seed=0,
)


# ----------------------------
# Student Policy
# ----------------------------
class StudentPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
            nn.Tanh(),  # Actions bounded [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ----------------------------
# Load teacher
# ----------------------------
def load_teacher(env):
    from stable_baselines3 import PPO
    teacher = PPO.load(config["teacher_path"], env=env, device=config["device"])
    teacher.policy.eval()
    return teacher


# ----------------------------
# Training (Distillation)
# ----------------------------
def train_student(student, teacher, env, run):
    optimizer = optim.Adam(student.parameters(), lr=config["lr"])
    n_steps_per_epoch = 10_000

    for epoch in range(config["n_epochs"]):
        obs = env.reset()  # VecEnv returns just the observation array
        obs_buf, act_teacher_buf = [], []

        for _ in range(n_steps_per_epoch):
            # Teacher action (predict expects vectorized obs)
            with torch.no_grad():
                teacher_action, _ = teacher.predict(obs, deterministic=True)

            # ---- FIX: ensure proper shape ----
            # teacher.predict returns actions for vectorized env, shape (n_envs, action_dim)
            # We need to extract the first (and only) environment's action
            if len(teacher_action.shape) == 1:
                # Already in correct shape (action_dim,)
                teacher_action = np.array(teacher_action, dtype=np.float32)
            else:
                # Shape is (n_envs, action_dim), extract first env
                teacher_action = np.array(teacher_action[0], dtype=np.float32)
            
            if teacher_action.shape[0] != env.action_space.shape[0]:
                raise ValueError(
                    f"Teacher action shape mismatch: got {teacher_action.shape}, expected {env.action_space.shape}"
                )

            # Collect data (flatten obs for single env)
            obs_flat = obs.reshape(-1) if len(obs.shape) > 1 else obs
            obs_buf.append(obs_flat.copy())
            act_teacher_buf.append(teacher_action.copy())

            # Step environment (VecEnv.step expects shape (n_envs, action_dim))
            teacher_action_vec = teacher_action.reshape(1, -1)
            # obs, _, terminated, truncated, _ = env.step(teacher_action_vec)
            obs, _, terminated, _ = env.step(teacher_action_vec)
            # Check for done (VecEnv returns arrays)
            done = bool(terminated[0])
            if done:
                obs = env.reset()

        # Build dataset
        obs_arr = np.array(obs_buf, dtype=np.float32)
        act_teacher_arr = np.array(act_teacher_buf, dtype=np.float32)

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(obs_arr, dtype=torch.float32),
            torch.tensor(act_teacher_arr, dtype=torch.float32),
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

        # Train student on this batch
        total_loss = 0.0
        student.train()
        for xb, yb in loader:
            xb, yb = xb.to(config["device"]), yb.to(config["device"])
            pred = student(xb)
            loss = F.mse_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        wandb.log({"student/train_loss": avg_loss, "epoch": epoch})
        print(f"[Epoch {epoch+1}] Loss = {avg_loss:.6f}")

    os.makedirs("models/student", exist_ok=True)
    torch.save(student.state_dict(), "models/student/student_policy.pth")
    wandb.save("models/student/student_policy.pth")
    print("✅ Student training complete.")


# ----------------------------
# Evaluation
# ----------------------------
def evaluate_student(student, env, n_episodes=5):
    student.eval()
    rewards = []

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        total_r = 0.0

        while not done:
            with torch.no_grad():
                obs_flat = obs.reshape(-1) if len(obs.shape) > 1 else obs
                obs_t = torch.tensor(obs_flat, dtype=torch.float32, device=config["device"]).unsqueeze(0)
                act = student(obs_t).cpu().numpy().reshape(1, -1)
            
            obs, r, terminated, _ = env.step(act)
            total_r += float(r[0])
            done = bool(terminated[0])

        rewards.append(total_r)
        print(f"Episode {ep+1}: Reward = {total_r:.2f}")

    mean_r = np.mean(rewards)
    wandb.log({"student/eval_mean_reward": mean_r})
    print(f"✅ Mean Eval Reward: {mean_r:.2f}")


# ----------------------------
# Main
# ----------------------------
def main():
    run = wandb.init(
        project=config["project"],
        name=f"DistillStudent_{int(time.time())}",
        config=config,
        save_code=True,
    )

    env = make_vec_env(config["env_name"], n_envs=1, vec_env_cls=DummyVecEnv, seed=config["seed"])
    if os.path.exists(config["vecnorm_path"]):
        env = VecNormalize.load(config["vecnorm_path"], env)
        env.training = False
        env.norm_reward = False

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    teacher = load_teacher(env)
    student = StudentPolicy(obs_dim, act_dim).to(config["device"])

    train_student(student, teacher, env, run)
    evaluate_student(student, env)

    run.finish()


if __name__ == "__main__":
    main()