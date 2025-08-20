#!/usr/bin/env python3
"""
Mini-Project: Contrastive Goal-Conditioned RL with TD3 + HER + Auto-Curriculum

This is a SINGLE-FILE reference implementation designed to run on any Gymnasium
GoalEnv-compatible task (dict observations with keys: 'observation',
'achieved_goal', 'desired_goal'). It has been tested primarily with
`gymnasium-robotics`' FetchPickAndPlace-v2 (MuJoCo) but the code is generic.

Core features:
- TD3 agent (actor-critic, target networks, policy delay, target policy smoothing)
- HER replay buffer (future strategy)
- Contrastive encoder f(s), f(g) trained with InfoNCE on replay to produce
  goal-aware embeddings; policy consumes [f(s) || f(g) || proprio]
- Optional dense shaping from state-goal L2 (can be toggled)
- Auto-curriculum that adjusts a scalar difficulty by calling an optional
  `set_difficulty(float)` method on the underlying env if available; otherwise
  it no-ops but still tracks/prints success-based difficulty.

Dependencies:
  pip install gymnasium gymnasium-robotics torch numpy

Example run:
  python gcrl_td3_her_contrastive_curriculum.py --env FetchPickAndPlace-v2 --steps 2_000_00 \
      --contrastive --auto_curriculum --dense_reward

Notes:
- If your environment isn't a GoalEnv, wrap it or adapt the observation parsing
  in `split_obs()`.
- If your env supports curriculum, expose `env.unwrapped.set_difficulty(x: float)`.
  This script will call it when success rate crosses thresholds.
- Keep this as a learning/reference script; for paper-quality results, move the
  classes into modules and add proper logging/ckpting.
"""

from __future__ import annotations
import argparse
import collections
import dataclasses
import math
import os
import random
import time
from typing import Dict, Tuple, Optional, List

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import gymnasium_robotics

gym.register_envs(gymnasium_robotics)
# -------------------- Utils --------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mlp(sizes, act=nn.ReLU, out_act=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
        else:
            layers.append(out_act())
    return nn.Sequential(*layers)


# -------------------- Networks --------------------

class Actor(nn.Module):
    def __init__(self, in_dim: int, act_dim: int, max_action: float, hidden=(256, 256)):
        super().__init__()
        self.net = mlp([in_dim, *hidden, act_dim])
        self.max_action = max_action

    def forward(self, x):
        return torch.tanh(self.net(x)) * self.max_action


class Critic(nn.Module):
    def __init__(self, in_dim: int, act_dim: int, hidden=(256, 256)):
        super().__init__()
        self.q1 = mlp([in_dim + act_dim, *hidden, 1])
        self.q2 = mlp([in_dim + act_dim, *hidden, 1])

    def forward(self, x, a):
        xa = torch.cat([x, a], dim=-1)
        return self.q1(xa), self.q2(xa)

    def q1_only(self, x, a):
        xa = torch.cat([x, a], dim=-1)
        return self.q1(xa)


# -------------------- Contrastive Encoder --------------------

class ContrastiveEncoder(nn.Module):
    """Encodes state and goal vectors into a shared embedding space.

    Use simple MLPs with shared trunk or Siamese-style weight sharing.
    We treat inputs as flat vectors (e.g., proprio+positions). If you have images,
    replace with a Conv encoder and feed flattened features here.
    """
    def __init__(self, in_dim_state: int, in_dim_goal: int, emb_dim: int = 64, hidden=(128, 128)):
        super().__init__()
        # Separate encoders allow asymmetric preprocessing if needed.
        self.state_enc = mlp([in_dim_state, *hidden, emb_dim])
        self.goal_enc = mlp([in_dim_goal, *hidden, emb_dim])

    def encode_state(self, s):
        z = self.state_enc(s)
        return F.normalize(z, dim=-1)

    def encode_goal(self, g):
        z = self.goal_enc(g)  
        return F.normalize(z, dim=-1)

    def info_nce_loss(self, z_s: torch.Tensor, z_g_pos: torch.Tensor, z_g_neg: torch.Tensor, temperature: float = 0.07):
        """InfoNCE with multiple negatives per state.
        z_s: [B, D], z_g_pos: [B, D], z_g_neg: [B, K, D]
        """
        B, D = z_s.shape
        K = z_g_neg.shape[1]
        # logits: [B, 1+K]
        pos = torch.sum(z_s * z_g_pos, dim=-1, keepdim=True) / temperature
        neg = torch.bmm(z_g_neg, z_s.unsqueeze(-1)).squeeze(-1) / temperature  # [B, K]
        logits = torch.cat([pos, neg], dim=1)
        labels = torch.zeros(B, dtype=torch.long, device=z_s.device)
        loss = F.cross_entropy(logits, labels)
        return loss


# -------------------- HER Replay Buffer --------------------

Transition = collections.namedtuple(
    "Transition",
    ["obs", "ag", "g", "action", "reward", "obs_next", "ag_next", "done"],
)

class HERBuffer:
    def __init__(self, obs_dim, ag_dim, g_dim, act_dim, size=int(1e6), her_k=4, future_p=0.8, device="cpu"):
        self.size = size
        self.her_k = her_k
        self.future_p = future_p
        self.device = device
        self.ptr = 0
        self.full = False
        # store as numpy for memory
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.ag = np.zeros((size, ag_dim), dtype=np.float32)
        self.g = np.zeros((size, g_dim), dtype=np.float32)
        self.actions = np.zeros((size, act_dim), dtype=np.float32)
        self.rewards = np.zeros((size, 1), dtype=np.float32)
        self.obs_next = np.zeros((size, obs_dim), dtype=np.float32)
        self.ag_next = np.zeros((size, ag_dim), dtype=np.float32)
        self.dones = np.zeros((size, 1), dtype=np.float32)
        # episode indices to support future relabeling
        self.ep_start_idxs: List[int] = []
        self.ep_lengths: List[int] = []
        self._current_ep_len = 0

    def start_episode(self):
        if self._current_ep_len > 0:
            self.ep_lengths.append(self._current_ep_len)
        self.ep_start_idxs.append(self.ptr)
        self._current_ep_len = 0

    def end_episode(self):
        if self._current_ep_len > 0:
            self.ep_lengths.append(self._current_ep_len)
            self._current_ep_len = 0

    def store(self, obs, ag, g, action, reward, obs_next, ag_next, done):
        idx = self.ptr
        self.obs[idx] = obs
        self.ag[idx] = ag
        self.g[idx] = g
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.obs_next[idx] = obs_next
        self.ag_next[idx] = ag_next
        self.dones[idx] = done

        self.ptr = (self.ptr + 1) % self.size
        self.full = self.full or self.ptr == 0
        self._current_ep_len += 1

    def _sample_idxs(self, batch_size):
        max_idx = self.size if self.full else self.ptr
        idxs = np.random.randint(0, max_idx, size=batch_size)
        return idxs

    def sample(self, batch_size, her_ratio=0.8, reward_fn=None):
        idxs = self._sample_idxs(batch_size)
        obs = self.obs[idxs].copy()
        ag = self.ag[idxs].copy()
        g = self.g[idxs].copy()
        actions = self.actions[idxs].copy()
        obs_next = self.obs_next[idxs].copy()
        ag_next = self.ag_next[idxs].copy()
        dones = self.dones[idxs].copy()

        # HER relabeling with future strategy
        her_mask = np.random.rand(batch_size) < her_ratio
        for i, use_her in enumerate(her_mask):
            if not use_her:
                continue
            # choose a future transition within same episode
            idx = idxs[i]
            # find episode bounds containing idx
            # naive scan (okay for clarity); for scale, precompute mapping
            ep_start, ep_end = None, None
            for s, l in zip(self.ep_start_idxs, self.ep_lengths):
                if s <= idx < s + l:
                    ep_start, ep_end = s, s + l
                    break
            if ep_start is None:
                continue
            future_idx = np.random.randint(idx, ep_end)
            g[i] = self.ag_next[future_idx]
            # recompute reward if fn given
        if reward_fn is not None:
            rewards = reward_fn(ag_next, g)
        else:
            rewards = self.rewards[idxs].copy()

        # to torch
        device = self.device
        to_t = lambda x: torch.as_tensor(x, device=device, dtype=torch.float32)
        batch = {
            'obs': to_t(obs), 'ag': to_t(ag), 'g': to_t(g), 'actions': to_t(actions),
            'obs_next': to_t(obs_next), 'ag_next': to_t(ag_next), 'dones': to_t(dones),
            'rewards': to_t(rewards)
        }
        return batch


# -------------------- TD3 Agent --------------------

class TD3:
    def __init__(self, state_dim, action_dim, max_action, lr=1e-3, gamma=0.98, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2, device="cpu"):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0
        self.device = device

    def select_action(self, state: np.ndarray, noise_std: float = 0.1):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]
        if noise_std > 0:
            action = action + np.random.normal(0, noise_std, size=action.shape)
        return np.clip(action, -self.max_action, self.max_action)

    def train(self, replay: HERBuffer, batch_size: int, reward_fn=None, actor_input_is_emb=True):
        self.total_it += 1
        batch = replay.sample(batch_size, reward_fn=reward_fn)
        # Construct policy inputs: either raw [obs||g] or embeddings already provided
        # In this single-file design, we assume upstream concatenation was done and passed in 'obs'
        # so we use obs as state input for actor/critic.
        state = batch['obs']
        next_state = batch['obs_next']
        action = batch['actions']
        reward = batch['rewards']
        done = batch['dones']

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target = reward + (1 - done) * self.gamma * target_q

        # Critic update
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Delayed actor update
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.q1_only(state, self.actor(state)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # Soft update
            with torch.no_grad():
                for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
                    p_targ.data.mul_(1 - self.tau)
                    p_targ.data.add_(self.tau * p.data)
                for p, p_targ in zip(self.actor.parameters(), self.actor_target.parameters()):
                    p_targ.data.mul_(1 - self.tau)
                    p_targ.data.add_(self.tau * p.data)

        return critic_loss.item()


# -------------------- Environment helpers --------------------

def split_obs(obs_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split GoalEnv dict observation into flat vectors.
    Returns (obs, achieved_goal, desired_goal) all as 1D np arrays.
    """
    return obs_dict["observation"].astype(np.float32), obs_dict["achieved_goal"].astype(np.float32), obs_dict["desired_goal"].astype(np.float32)


def default_reward_fn(ag_next: np.ndarray, g: np.ndarray) -> np.ndarray:
    # Negative distance + success bonus can be implemented in env; here we keep simple dense shaping
    d = np.linalg.norm(ag_next - g, axis=-1, keepdims=True)
    return -d


# -------------------- Auto-Curriculum --------------------

@dataclasses.dataclass
class Curriculum:
    target_high: float = 0.7
    target_low: float = 0.3
    difficulty: float = 0.2  # [0,1]
    step: float = 0.05
    window: int = 100
    history: collections.deque = dataclasses.field(default_factory=lambda: collections.deque(maxlen=100))

    def update(self, success: bool, env=None):
        self.history.append(1.0 if success else 0.0)
        if len(self.history) < max(10, self.history.maxlen // 5):
            return self.difficulty
        rate = np.mean(self.history)
        if rate > self.target_high:
            self.difficulty = min(1.0, self.difficulty + self.step)
        elif rate < self.target_low:
            self.difficulty = max(0.0, self.difficulty - self.step)
        # Try to notify env if it supports difficulty
        if env is not None and hasattr(env.unwrapped, "set_difficulty"):
            try:
                env.unwrapped.set_difficulty(float(self.difficulty))
            except Exception:
                pass
        return self.difficulty


# -------------------- Training Loop --------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_seed(args.seed)

    env = gym.make(args.env)
    eval_env = gym.make(args.env)

    assert isinstance(env.observation_space, gym.spaces.Dict), "Env must be GoalEnv-like with Dict observations"

    obs0, _ = env.reset(seed=args.seed)
    o, ag, g = split_obs(obs0)

    obs_dim = o.shape[0]
    ag_dim = ag.shape[0]
    g_dim = g.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Contrastive encoder
    enc = None
    emb_dim = 0
    if args.contrastive:
        enc = ContrastiveEncoder(in_dim_state=obs_dim + ag_dim, in_dim_goal=g_dim, emb_dim=args.emb_dim).to(device)
        enc_opt = torch.optim.Adam(enc.parameters(), lr=args.enc_lr)
        emb_dim = args.emb_dim

    # TD3 over embedding input
    policy_in_dim = (emb_dim * 2 + obs_dim) if args.contrastive else (obs_dim + g_dim)
    agent = TD3(policy_in_dim, act_dim, max_action, lr=args.lr, gamma=args.gamma, tau=args.tau,
                policy_noise=args.policy_noise, noise_clip=args.noise_clip, policy_freq=args.policy_freq, device=device)

    # Replay
    replay = HERBuffer(obs_dim=policy_in_dim, ag_dim=ag_dim, g_dim=g_dim, act_dim=act_dim, size=args.buffer_size,
                       her_k=args.her_k, future_p=args.future_p, device=device)

    # Curriculum
    curriculum = Curriculum(window=args.cur_window)

    # For reward shaping
    if args.dense_reward:
        reward_fn = default_reward_fn
    else:
        reward_fn = None

    ep_ret, ep_len, ep_success = 0.0, 0, False
    success_stats = collections.deque(maxlen=100)

    # Start first episode in buffer (for episode-boundaries used by HER)
    replay.start_episode()

    obs_dict, _ = env.reset(seed=args.seed)
    o, ag, g = split_obs(obs_dict)

    # Helper to build policy state input
    def build_state(o_vec, ag_vec, g_vec):
        o_t = torch.as_tensor(o_vec, dtype=torch.float32, device=device).unsqueeze(0)
        ag_t = torch.as_tensor(ag_vec, dtype=torch.float32, device=device).unsqueeze(0)
        g_t = torch.as_tensor(g_vec, dtype=torch.float32, device=device).unsqueeze(0)
        if args.contrastive:
            with torch.no_grad():
                z_s = enc.encode_state(torch.cat([o_t, ag_t], dim=-1))
                z_g = enc.encode_goal(g_t)
            state = torch.cat([z_s, z_g, o_t], dim=-1).squeeze(0).cpu().numpy()
        else:
            state = np.concatenate([o_vec, g_vec], axis=-1)
        return state

    state = build_state(o, ag, g)

    # Storage for contrastive negatives
    goal_bank = collections.deque(maxlen=10_000)

    for t in range(1, args.steps + 1):
        # Select action
        if t < args.start_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, noise_std=args.expl_noise)

        # Step
        obs_next_dict, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        o2, ag2, g2 = split_obs(obs_next_dict)

        # Compute reward if dense shaping is on (for buffer training); env may also return sparse reward via info
        if args.dense_reward:
            r = float(-np.linalg.norm(ag2 - g2))
        else:
            # try to read env's reward function via compute_reward if available
            if hasattr(env.unwrapped, 'compute_reward'):
                r = float(env.unwrapped.compute_reward(ag2, g2, info))
            else:
                r = 0.0

        state_next = build_state(o2, ag2, g2)

        # Store transition
        replay.store(state, ag, g, action, r, state_next, ag2, float(done))

        ep_len += 1
        ep_ret += r

        # success flag if provided
        if 'is_success' in info:
            ep_success = bool(info['is_success'])

        # Move on
        state = state_next
        o, ag, g = o2, ag2, g2

        if done:
            success_stats.append(1.0 if ep_success else 0.0)
            if args.auto_curriculum:
                curriculum.update(ep_success, env)
            # reset
            obs_dict, _ = env.reset()
            o, ag, g = split_obs(obs_dict)
            state = build_state(o, ag, g)
            replay.end_episode()
            replay.start_episode()
            ep_ret, ep_len, ep_success = 0.0, 0, False

        # Fill negative goal bank
        if len(goal_bank) < goal_bank.maxlen:
            goal_bank.append(g.copy())

        # Updates
        if t >= args.update_after and t % args.update_every == 0:
            for _ in range(args.update_every):
                # Contrastive pretrain / joint train
                if args.contrastive:
                    batch = replay.sample(args.batch_size, reward_fn=reward_fn)
                    # Build positives: (s: [o||ag], g: goal used in that transition)
                    s_pos = torch.cat([batch['obs'][:, -obs_dim:], batch['ag']], dim=-1)  # Recover o from tail
                    g_pos = batch['g']
                    z_s = enc.encode_state(s_pos)
                    z_g_pos = enc.encode_goal(g_pos)
                    # Negatives: sample K negatives per state
                    K = args.n_negatives
                    with torch.no_grad():
                        neg_goals = []
                        for _ in range(K):
                            idxs = np.random.randint(0, len(goal_bank), size=s_pos.shape[0])
                            g_neg_np = np.stack([goal_bank[i] for i in idxs], axis=0)
                            neg_goals.append(torch.as_tensor(g_neg_np, device=device, dtype=torch.float32))
                        z_g_neg_list = [enc.encode_goal(g_neg) for g_neg in neg_goals]
                        z_g_neg = torch.stack(z_g_neg_list, dim=1)  # [B,K,D]
                    loss_c = enc.info_nce_loss(z_s, z_g_pos, z_g_neg, temperature=args.temperature)
                    enc_opt.zero_grad()
                    loss_c.backward()
                    enc_opt.step()

                # TD3 update
                agent.train(replay, batch_size=args.batch_size, reward_fn=reward_fn)

        # Periodic eval
        if t % args.eval_every == 0:
            sr = np.mean(success_stats) if len(success_stats) else 0.0
            diff = curriculum.difficulty if args.auto_curriculum else None
            print(f"Step {t:>8} | recent SR={sr:.2f} | diff={diff} | replay_ptr={replay.ptr} | device={device}")

    env.close()
    eval_env.close()


# -------------------- CLI --------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--env', type=str, default='FetchPickAndPlace-v3', help='Any GoalEnv-like env id')
    p.add_argument('--steps', type=int, default=200_000)
    p.add_argument('--start_steps', type=int, default=5000)
    p.add_argument('--update_after', type=int, default=5000)
    p.add_argument('--update_every', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--buffer_size', type=int, default=1_000_000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--gamma', type=float, default=0.98)
    p.add_argument('--tau', type=float, default=0.005)
    p.add_argument('--policy_noise', type=float, default=0.2)
    p.add_argument('--noise_clip', type=float, default=0.5)
    p.add_argument('--policy_freq', type=int, default=2)
    p.add_argument('--expl_noise', type=float, default=0.1)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--seed', type=int, default=42)

    # Contrastive
    p.add_argument('--contrastive', action='store_true')
    p.add_argument('--emb_dim', type=int, default=64)
    p.add_argument('--enc_lr', type=float, default=1e-3)
    p.add_argument('--n_negatives', type=int, default=16)
    p.add_argument('--temperature', type=float, default=0.07)

    # Reward shaping
    p.add_argument('--dense_reward', action='store_true')

    # Auto-curriculum
    p.add_argument('--auto_curriculum', action='store_true')
    p.add_argument('--cur_window', type=int, default=100)

    # Eval/prints
    p.add_argument('--eval_every', type=int, default=5000)

    p.add_argument("--her_k", type=int, default=4, help="Number of HER replays per real transition")
    p.add_argument("--future_p", type=float, default=0.8, help="Probability of sampling a future goal for HER")

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
