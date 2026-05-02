# requires: torch numpy matplotlib
"""
J27 - Capstone : eval, ablations, baseline BC.

Goal: produce the canonical eval table for a Diffusion-Policy-style capstone
without depending on the J24-J26 artifacts. We build a self-contained pipeline:

  1. A toy 2D PushT-like environment (block + target, push action).
  2. Mock policies:
       - DiffusionPolicy : action chunk + DDPM-style "denoising" + EMA.
       - DiffusionPolicy ablation 1 : no chunking (H=1).
       - DiffusionPolicy ablation 2 : no EMA.
       - BehaviorCloning : direct (obs -> action) MLP.
  3. Evaluation protocol: K seeds x N rollouts per seed, with metrics
       success_rate, episode_length, action_smoothness, latency_ms.
  4. Comparative table (markdown printed) + matplotlib bar plot saved to disk.

Self-contained on purpose: this should run as
    python domains/robotics-ai/02-code/27-capstone-eval-ablations.py
without any J24-J26 file present. Comments below show how to swap in the
real J26 checkpoint and the J24 PushT environment for production usage.

Source: REFERENCES.md #19, Chi et al., "Diffusion Policy: Visuomotor Policy
Learning via Action Diffusion", RSS 2023 / IJRR 2024, section 6 (Experiments).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

# matplotlib import is guarded: we still want py_compile to pass even on a
# minimal environment, and we only call plt at the very end.
try:
    import matplotlib

    matplotlib.use("Agg")  # headless for CI / subagent execution
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:
    HAS_MPL = False


# ----------------------------------------------------------------------------
# 1. Toy 2D PushT-like environment
# ----------------------------------------------------------------------------
# We mimic the *spirit* of PushT (Diffusion Policy benchmark): a 2D arena where
# an "agent" (cursor) must push a "block" toward a "target". State is fully
# observable as a 6D vector (cursor_xy, block_xy, target_xy). Action is a 2D
# delta cursor velocity in [-1, 1]. Success = block within tolerance of target.
#
# In production you would replace this with the J24 LeRobot PushT env:
#     env = gym.make("PushT-v0")  # or the lerobot-wrapped variant


@dataclass
class EnvConfig:
    arena: float = 1.0  # arena half-size; block/agent/target live in [-arena, arena]^2
    horizon: int = 200  # max steps per episode
    success_tol: float = 0.05  # block within 0.05 of target = success
    push_strength: float = 0.6  # how much agent_velocity transfers to block when in contact
    contact_radius: float = 0.08  # agent within this distance "pushes" the block
    dt: float = 1.0 / 10  # 10 Hz control


class ToyPushT:
    """Toy 2D push-to-target environment, reproducible via seed."""

    def __init__(self, cfg: EnvConfig | None = None):
        self.cfg = cfg or EnvConfig()
        self.rng = np.random.default_rng(0)
        self._reset_state()

    def _reset_state(self):
        c = self.cfg
        # Initial state randomized in arena, with a minimum distance constraint
        self.agent = self.rng.uniform(-c.arena, c.arena, size=2).astype(np.float32)
        self.block = self.rng.uniform(-c.arena * 0.7, c.arena * 0.7, size=2).astype(np.float32)
        self.target = self.rng.uniform(-c.arena * 0.7, c.arena * 0.7, size=2).astype(np.float32)
        # Avoid degenerate "block already on target" inits
        while np.linalg.norm(self.block - self.target) < 0.3:
            self.target = self.rng.uniform(-c.arena * 0.7, c.arena * 0.7, size=2).astype(np.float32)
        self.t = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_state()
        return self._obs()

    def _obs(self) -> np.ndarray:
        return np.concatenate([self.agent, self.block, self.target]).astype(np.float32)

    def step(self, action: np.ndarray):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        c = self.cfg
        # Move agent
        self.agent = np.clip(self.agent + a * c.dt, -c.arena, c.arena)
        # Push the block if agent is close enough; very simple "contact" model
        d = self.agent - self.block
        dist = np.linalg.norm(d) + 1e-8
        if dist < c.contact_radius:
            push_dir = -d / dist  # block moves *away* from agent
            self.block = np.clip(
                self.block + push_dir * c.push_strength * c.dt,
                -c.arena, c.arena,
            )
        self.t += 1
        success = np.linalg.norm(self.block - self.target) < c.success_tol
        terminated = bool(success)
        truncated = self.t >= c.horizon
        reward = 1.0 if success else 0.0
        return self._obs(), reward, terminated, truncated, {"success": success}


# ----------------------------------------------------------------------------
# 2. Expert oracle for synthetic dataset generation
# ----------------------------------------------------------------------------
# The expert "pretends" to be a perfect controller: it computes the direction
# from agent->block, then block->target, and pushes accordingly.
# In production this would be replaced by J24 teleop demos in LeRobotDataset.


def expert_action(obs: np.ndarray) -> np.ndarray:
    agent, block, target = obs[:2], obs[2:4], obs[4:6]
    # First go behind the block (opposite side from target), then push.
    push_dir = (target - block)
    push_norm = np.linalg.norm(push_dir) + 1e-8
    push_dir = push_dir / push_norm
    behind_block = block - push_dir * 0.06  # small offset behind the block
    delta_to_behind = behind_block - agent
    if np.linalg.norm(delta_to_behind) > 0.04:
        # Approach phase
        a = delta_to_behind / (np.linalg.norm(delta_to_behind) + 1e-8)
    else:
        # Push phase
        a = push_dir
    return np.clip(a, -1.0, 1.0).astype(np.float32)


def collect_demos(n_episodes: int, horizon: int, seed: int = 0):
    """Roll the expert and return (obs, actions) flattened across episodes."""
    env = ToyPushT(EnvConfig(horizon=horizon))
    obs_buf, act_buf = [], []
    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        for _ in range(horizon):
            a = expert_action(obs)
            obs_buf.append(obs.copy())
            act_buf.append(a.copy())
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
    return np.stack(obs_buf), np.stack(act_buf)


# ----------------------------------------------------------------------------
# 3. Policies
# ----------------------------------------------------------------------------
# All three policies share: obs_dim=6, action_dim=2.
# They differ in:
#   - DiffusionPolicy: predicts an action chunk of length H, conditioned on obs,
#     with a small "denoiser" that does T_denoise iterative refinement steps.
#     Trained with noise-prediction loss (DDPM style, simplified).
#   - BC: predicts 1 action from 1 obs (regression).


class MLPDenoiser(nn.Module):
    """Simplified UNet-1D-like denoiser. Input: noised action chunk + obs + t.
    Output: predicted noise of same shape as the chunk.
    """

    def __init__(self, obs_dim: int, action_dim: int, horizon: int, hidden: int = 128):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        in_dim = horizon * action_dim + obs_dim + 1  # +1 for normalized timestep t
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, horizon * action_dim),
        )

    def forward(self, x_t: torch.Tensor, obs: torch.Tensor, t_norm: torch.Tensor) -> torch.Tensor:
        B = x_t.shape[0]
        flat = x_t.reshape(B, -1)
        h = torch.cat([flat, obs, t_norm.unsqueeze(-1)], dim=-1)
        out = self.net(h).reshape(B, self.horizon, self.action_dim)
        return out


class DiffusionPolicy:
    """Pedagogical Diffusion Policy. Uses simplified DDPM with linear schedule.

    Args mapped to ablations:
        horizon: 16 = full chunking ; 1 = ablation "no chunking".
        use_ema: True = full ; False = ablation "no EMA".
    """

    def __init__(
        self,
        obs_dim: int = 6,
        action_dim: int = 2,
        horizon: int = 16,
        T_denoise: int = 50,
        use_ema: bool = True,
        ema_decay: float = 0.995,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.T_denoise = T_denoise
        self.use_ema = use_ema
        self.device = device

        self.model = MLPDenoiser(obs_dim, action_dim, horizon).to(device)
        self.ema_model = MLPDenoiser(obs_dim, action_dim, horizon).to(device) if use_ema else None
        if self.ema_model is not None:
            self.ema_model.load_state_dict(self.model.state_dict())
        self.ema_decay = ema_decay

        # Linear beta schedule, classic DDPM (Ho 2020).
        betas = torch.linspace(1e-4, 0.02, T_denoise)
        alphas = 1 - betas
        self.alphas_cum = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas

        # For receding horizon eval
        self.cached_chunk: np.ndarray | None = None
        self.cached_idx: int = 0

    # ---- training helpers ----
    def _q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        eps = torch.randn_like(x0)
        a_bar = self.alphas_cum[t].view(-1, 1, 1)
        x_t = a_bar.sqrt() * x0 + (1 - a_bar).sqrt() * eps
        return x_t, eps

    def _ema_update(self):
        if not self.use_ema or self.ema_model is None:
            return
        with torch.no_grad():
            for p, p_ema in zip(self.model.parameters(), self.ema_model.parameters()):
                p_ema.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)

    def fit(self, obs: np.ndarray, actions: np.ndarray, n_steps: int = 800, lr: float = 3e-4, batch: int = 64):
        """Train on (obs, action_chunk) pairs reconstructed by sliding window.

        Trick: we transform (obs[t], action[t]) sequences into chunks of length
        self.horizon, padding with the last action when needed.
        """
        H = self.horizon
        # Build chunks: for each t, take actions[t : t+H] (clipped at end of buffer).
        N = len(actions)
        chunks = np.zeros((N, H, self.action_dim), dtype=np.float32)
        for i in range(N):
            for k in range(H):
                idx = min(i + k, N - 1)
                chunks[i, k] = actions[idx]
        obs_t = torch.from_numpy(obs).float().to(self.device)
        chunks_t = torch.from_numpy(chunks).float().to(self.device)

        opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        for step in range(n_steps):
            idx = np.random.randint(0, N, size=batch)
            o = obs_t[idx]
            x0 = chunks_t[idx]
            t = torch.randint(0, self.T_denoise, (batch,))
            x_t, eps = self._q_sample(x0, t)
            t_norm = t.float() / self.T_denoise
            eps_pred = self.model(x_t, o, t_norm)
            loss = ((eps_pred - eps) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            self._ema_update()

    # ---- inference (DDPM ancestral sampling) ----
    @torch.no_grad()
    def _sample_chunk(self, obs_np: np.ndarray) -> np.ndarray:
        model = self.ema_model if (self.use_ema and self.ema_model is not None) else self.model
        model.eval()
        obs = torch.from_numpy(obs_np).float().unsqueeze(0).to(self.device)
        x = torch.randn(1, self.horizon, self.action_dim, device=self.device)
        for t in reversed(range(self.T_denoise)):
            t_norm = torch.tensor([t / self.T_denoise], device=self.device)
            eps = model(x, obs, t_norm)
            a = self.alphas[t]
            a_bar = self.alphas_cum[t]
            x = (1.0 / a.sqrt()) * (x - (1 - a) / (1 - a_bar).sqrt() * eps)
            if t > 0:
                noise = torch.randn_like(x)
                sigma = self.betas[t].sqrt()
                x = x + sigma * noise
        return x.squeeze(0).cpu().numpy().astype(np.float32)

    def predict(self, obs: np.ndarray, T_alpha: int = 8) -> np.ndarray:
        """Receding-horizon: cache a chunk, replan after T_alpha actions.

        T_alpha = 8 (default) follows §6.3 of the Diffusion Policy paper.
        For ablation H=1, T_alpha is forced to 1 by construction.
        """
        T_alpha = min(T_alpha, self.horizon)
        if (self.cached_chunk is None) or (self.cached_idx >= T_alpha):
            self.cached_chunk = self._sample_chunk(obs)
            self.cached_idx = 0
        action = self.cached_chunk[self.cached_idx]
        self.cached_idx += 1
        return action

    def reset_cache(self):
        self.cached_chunk = None
        self.cached_idx = 0


class BCPolicy:
    """Behavior cloning: a small MLP regressing actions from observations.

    Trained with MSE loss on (obs, action) pairs from the expert dataset.
    """

    def __init__(self, obs_dim: int = 6, action_dim: int = 2, hidden: int = 128, device: str = "cpu"):
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, action_dim), nn.Tanh(),
        ).to(device)
        self.action_dim = action_dim

    def fit(self, obs: np.ndarray, actions: np.ndarray, n_steps: int = 800, lr: float = 3e-4, batch: int = 64):
        obs_t = torch.from_numpy(obs).float().to(self.device)
        act_t = torch.from_numpy(actions).float().to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr)
        for step in range(n_steps):
            idx = np.random.randint(0, len(obs), size=batch)
            o, a = obs_t[idx], act_t[idx]
            pred = self.net(o)
            loss = ((pred - a) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    def predict(self, obs: np.ndarray, T_alpha: int = 1) -> np.ndarray:  # T_alpha unused, kept for API parity
        with torch.no_grad():
            o = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            a = self.net(o).squeeze(0).cpu().numpy().astype(np.float32)
        return a

    def reset_cache(self):
        pass


# ----------------------------------------------------------------------------
# 4. Evaluation protocol
# ----------------------------------------------------------------------------
# K seeds x N rollouts per seed, metrics aggregated as mean +/- std across seeds.


@dataclass
class EvalResult:
    name: str
    success_rate_mean: float
    success_rate_std: float
    episode_length: float
    action_smoothness: float
    latency_ms: float

    def as_row(self) -> str:
        return (
            f"{self.name:<32}"
            f"{self.success_rate_mean:.2f} +/- {self.success_rate_std:.2f}   "
            f"{self.episode_length:7.1f}   "
            f"{self.action_smoothness:7.4f}   "
            f"{self.latency_ms:7.2f}"
        )


def measure_latency(policy, obs_dim: int = 6, n_warmup: int = 5, n_measure: int = 30) -> float:
    """Per-step latency in ms, averaged over n_measure calls."""
    obs = np.zeros(obs_dim, dtype=np.float32)
    policy.reset_cache()
    # Warmup: trigger first inference (which is the most expensive for DDPM
    # because there is no cached chunk).
    for _ in range(n_warmup):
        policy.predict(obs)
    times = []
    for _ in range(n_measure):
        policy.reset_cache()
        t0 = time.perf_counter()
        policy.predict(obs)
        times.append(time.perf_counter() - t0)
    return 1000.0 * float(np.mean(times))


def rollout_one(env: ToyPushT, policy, T_alpha: int) -> dict:
    obs = env.reset()
    policy.reset_cache()
    actions_taken = []
    success = False
    ep_len = 0
    for t in range(env.cfg.horizon):
        a = policy.predict(obs, T_alpha=T_alpha) if hasattr(policy, "predict") else policy(obs)
        actions_taken.append(a)
        obs, _, term, trunc, info = env.step(a)
        ep_len += 1
        if info.get("success"):
            success = True
            break
        if trunc:
            break
    actions_taken = np.stack(actions_taken)
    if len(actions_taken) >= 2:
        smoothness = float(np.mean(np.sum((actions_taken[1:] - actions_taken[:-1]) ** 2, axis=-1)))
    else:
        smoothness = 0.0
    return {"success": success, "ep_len": ep_len, "smooth": smoothness}


def evaluate(policy, name: str, K_seeds: int = 3, N_rollouts: int = 20, T_alpha: int = 8) -> EvalResult:
    success_per_seed = []
    ep_lens, smooths = [], []
    for s in range(K_seeds):
        env = ToyPushT(EnvConfig())
        env.rng = np.random.default_rng(1000 + s)  # parent seed per run
        successes = 0
        for r in range(N_rollouts):
            # New env state per rollout, but deterministic across runs.
            env.rng = np.random.default_rng(1000 * (s + 1) + r)
            res = rollout_one(env, policy, T_alpha=T_alpha)
            if res["success"]:
                successes += 1
                ep_lens.append(res["ep_len"])
            smooths.append(res["smooth"])
        success_per_seed.append(successes / N_rollouts)
    sr_mean = float(np.mean(success_per_seed))
    sr_std = float(np.std(success_per_seed))
    avg_ep_len = float(np.mean(ep_lens)) if ep_lens else float(env.cfg.horizon)
    avg_smooth = float(np.mean(smooths))
    lat_ms = measure_latency(policy)
    return EvalResult(name, sr_mean, sr_std, avg_ep_len, avg_smooth, lat_ms)


# ----------------------------------------------------------------------------
# 5. Main: train policies, evaluate, print table, plot
# ----------------------------------------------------------------------------


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    print("[J27] Collecting expert demos (toy PushT)...")
    obs, actions = collect_demos(n_episodes=80, horizon=200, seed=42)
    print(f"  -> {len(obs)} (obs, action) transitions collected.")

    print("[J27] Training Diffusion Policy (full)...")
    dp_full = DiffusionPolicy(horizon=16, T_denoise=50, use_ema=True)
    dp_full.fit(obs, actions, n_steps=600)

    print("[J27] Training Diffusion Policy (ablation: no chunking, H=1)...")
    dp_no_chunk = DiffusionPolicy(horizon=1, T_denoise=50, use_ema=True)
    dp_no_chunk.fit(obs, actions, n_steps=600)

    print("[J27] Training Diffusion Policy (ablation: no EMA)...")
    dp_no_ema = DiffusionPolicy(horizon=16, T_denoise=50, use_ema=False)
    dp_no_ema.fit(obs, actions, n_steps=600)

    print("[J27] Training BC baseline (MLP)...")
    bc = BCPolicy()
    bc.fit(obs, actions, n_steps=600)

    print("[J27] Evaluating (K=3 seeds x N=20 rollouts each)...")
    K, N = 3, 20  # N=50-100 in the paper; we keep N=20 here so the script is fast on CPU
    results = []
    results.append(evaluate(dp_full, "Diffusion Policy (full)", K, N, T_alpha=8))
    results.append(evaluate(dp_no_chunk, "DP - no chunking (H=1)", K, N, T_alpha=1))
    results.append(evaluate(dp_no_ema, "DP - no EMA", K, N, T_alpha=8))
    results.append(evaluate(bc, "Behavior Cloning (MLP)", K, N, T_alpha=1))

    # ---- Print comparative table ----
    print()
    header = (
        f"{'Method':<32}"
        f"{'success_rate':<16}"
        f"{'ep_len':<10}"
        f"{'smooth':<10}"
        f"{'lat_ms':<8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(r.as_row())
    print()

    # ---- Markdown export ----
    print("Markdown:")
    print()
    print("| Method | Success | EpLen | Smooth | Latency (ms) |")
    print("|---|---|---|---|---|")
    for r in results:
        print(
            f"| {r.name} | {r.success_rate_mean:.2f} +/- {r.success_rate_std:.2f} "
            f"| {r.episode_length:.1f} | {r.action_smoothness:.4f} | {r.latency_ms:.1f} |"
        )

    # ---- Plot ----
    if HAS_MPL:
        names = [r.name.replace("Diffusion Policy", "DP") for r in results]
        sr = [r.success_rate_mean for r in results]
        sr_err = [r.success_rate_std for r in results]
        smooth = [r.action_smoothness for r in results]
        lat = [r.latency_ms for r in results]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        axes[0].bar(range(len(names)), sr, yerr=sr_err, color="steelblue")
        axes[0].set_xticks(range(len(names)))
        axes[0].set_xticklabels(names, rotation=20, ha="right")
        axes[0].set_ylabel("Success rate")
        axes[0].set_ylim(0, 1)
        axes[0].set_title("Success rate (mean +/- std across 3 seeds)")
        axes[1].bar(range(len(names)), smooth, color="indianred")
        axes[1].set_xticks(range(len(names)))
        axes[1].set_xticklabels(names, rotation=20, ha="right")
        axes[1].set_ylabel("Mean ||a_{t+1} - a_t||^2")
        axes[1].set_title("Action smoothness (lower is smoother)")
        axes[2].bar(range(len(names)), lat, color="seagreen")
        axes[2].set_xticks(range(len(names)))
        axes[2].set_xticklabels(names, rotation=20, ha="right")
        axes[2].set_ylabel("Latency (ms)")
        axes[2].set_title("Per-step inference latency")
        plt.tight_layout()
        out = "j27_eval_results.png"
        plt.savefig(out, dpi=120)
        print(f"\nPlot saved to: {out}")
    else:
        print("\n(matplotlib not available - plot skipped)")

    # ---- How to plug in real artifacts (J24 env + J26 checkpoint) ----
    # In production, replace the toy pieces above with:
    #
    #   import gymnasium as gym
    #   env = gym.make("PushT-v0")                          # J24 env
    #   policy = load_diffusion_policy("checkpoints/j26.pt")  # J26 checkpoint
    #   evaluate(policy, "DP J26 ckpt", K_seeds=3, N_rollouts=50, T_alpha=8)
    #
    # The eval/rollout/metrics logic (success_rate, ep_len, smoothness, latency)
    # is intentionally policy-agnostic and re-usable as-is.


if __name__ == "__main__":
    main()
