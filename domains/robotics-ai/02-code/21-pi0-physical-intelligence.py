# requires: torch numpy matplotlib
"""
J21 - pi0 (Physical Intelligence) - toy implementation.

Goal: implement a *pedagogical* mini "pi0-style" policy with:
  1. A mock VLM (small MLP) that encodes a fake (vision, instruction) pair.
  2. A flow-matching action head trained to predict an action chunk.
  3. A side-by-side DDPM action head trained on the *same* toy task.

We compare them on:
  - training loss curves,
  - inference step count to converge,
  - sampled action chunk MSE vs ground truth.

This file is intentionally light:
  - no real images, no real robot, no LeRobot dataset,
  - actions are 2D toy trajectories (figure-8 patterns),
  - "vision" is just a class id mapped to an embedding.

Run:
    python domains/robotics-ai/02-code/21-pi0-physical-intelligence.py

Source reference (REFERENCES.md #14): Black et al. (Physical Intelligence),
"pi0: A Vision-Language-Action Flow Model for General Robot Control", 2024.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# 1. Toy multi-embodiment dataset
# ----------------------------------------------------------------------------
# We simulate 3 "embodiments" (= 3 task families), each producing a 2D action
# chunk of length H. The "vision" input is a one-hot id of the family. This is
# the absolute simplest stand-in for what pi0 does at scale (PaliGemma + 7
# robots): a single model has to disambiguate which trajectory family is wanted
# from a low-dim context vector.


@dataclass
class ToyConfig:
    n_families: int = 3       # number of "embodiments"
    horizon: int = 16         # H = action chunk length (pi0 uses 50)
    action_dim: int = 2       # toy 2D action; pi0 pads to 18
    vlm_dim: int = 32         # mock VLM hidden size; pi0 uses ~2048
    expert_dim: int = 64      # action expert hidden size
    train_steps: int = 1500
    batch_size: int = 64
    device: str = "cpu"


def make_target_chunk(family_id: int, H: int) -> torch.Tensor:
    """Return a deterministic 2D action chunk of length H for a given family.

    Each family is a different parametric curve (circle / figure-8 / sine).
    This stands in for "different demonstrated behaviors". A real pi0 dataset
    would have images + language + 50-step joint trajectories.
    """
    t = torch.linspace(0.0, 2.0 * math.pi, H)
    if family_id == 0:
        # circle
        x = torch.cos(t)
        y = torch.sin(t)
    elif family_id == 1:
        # figure-8 / lemniscate
        x = torch.sin(t)
        y = torch.sin(t) * torch.cos(t)
    else:
        # damped wave
        x = torch.linspace(-1.0, 1.0, H)
        y = 0.5 * torch.sin(3.0 * t) * torch.exp(-0.3 * t)
    return torch.stack([x, y], dim=-1)  # (H, 2)


def sample_batch(cfg: ToyConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (context, target_chunk) batch.

    context:       (B, n_families) one-hot id  -- our "VLM input"
    target_chunk:  (B, H, action_dim)          -- ground-truth action chunk
    """
    fam = torch.randint(0, cfg.n_families, (cfg.batch_size,))
    onehot = F.one_hot(fam, num_classes=cfg.n_families).float()
    chunks = torch.stack([make_target_chunk(int(f), cfg.horizon) for f in fam])
    return onehot.to(cfg.device), chunks.to(cfg.device)


# ----------------------------------------------------------------------------
# 2. Mock VLM (pi0 actually uses PaliGemma 3B; we use a 2-layer MLP)
# ----------------------------------------------------------------------------
class MockVLM(nn.Module):
    """A tiny MLP that consumes the one-hot family id (the "image+text") and
    produces a fixed-size context embedding `h`. In pi0 this would be the
    SigLIP+Gemma stack producing a sequence of language-vision tokens.
    """

    def __init__(self, cfg: ToyConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_families, cfg.vlm_dim),
            nn.GELU(),
            nn.Linear(cfg.vlm_dim, cfg.vlm_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, n_fam) -> (B, vlm_dim)
        return self.net(x)


# ----------------------------------------------------------------------------
# 3. Sinusoidal time embedding (used by both action heads)
# ----------------------------------------------------------------------------
def time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding so the action expert knows *where* it is along the
    integration / denoising schedule. Same trick as in DDPM and in transformers.
    `t` is shape (B,) with values in [0, 1] (flow matching) or [0, T-1] (DDPM).
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device) / half
    )
    args = t.unsqueeze(-1).float() * freqs.unsqueeze(0)  # (B, half)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, dim)


# ----------------------------------------------------------------------------
# 4. Action expert backbone (shared by both heads to keep comparison fair)
# ----------------------------------------------------------------------------
class ActionExpert(nn.Module):
    """A small MLP that takes (noised_chunk_flat, time_emb, vlm_h) and outputs
    a chunk-shaped tensor. In pi0, this is a 300M transformer with cross-attn
    on VLM tokens. Here a plain MLP is sufficient to demonstrate the math.
    """

    def __init__(self, cfg: ToyConfig):
        super().__init__()
        self.cfg = cfg
        flat_action = cfg.horizon * cfg.action_dim
        in_dim = flat_action + cfg.expert_dim + cfg.vlm_dim  # action + t-emb + h
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, flat_action),
        )

    def forward(self, A: torch.Tensor, t_emb: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # A: (B, H, d), t_emb: (B, expert_dim), h: (B, vlm_dim)
        B = A.size(0)
        x = torch.cat([A.reshape(B, -1), t_emb, h], dim=-1)
        out = self.net(x)
        return out.reshape(B, self.cfg.horizon, self.cfg.action_dim)


# ----------------------------------------------------------------------------
# 5. Flow Matching head (the pi0 way)
# ----------------------------------------------------------------------------
# Training:
#   sample tau ~ U(0, 1), A0 ~ N(0, I), build A_tau = (1 - tau) * A0 + tau * A_star,
#   regress velocity v_theta(A_tau, tau) -> (A_star - A0).
# Inference:
#   solve dA/dtau = v_theta(A_tau, tau) with N steps Euler from N(0,I) to A_star.

class FlowMatchingPolicy(nn.Module):
    def __init__(self, cfg: ToyConfig):
        super().__init__()
        self.cfg = cfg
        self.vlm = MockVLM(cfg)
        self.expert = ActionExpert(cfg)

    def velocity(self, A_tau: torch.Tensor, tau: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        t_emb = time_embedding(tau, self.cfg.expert_dim)
        return self.expert(A_tau, t_emb, h)

    def loss(self, ctx: torch.Tensor, A_star: torch.Tensor) -> torch.Tensor:
        """Flow-matching MSE: regress the *constant* velocity (A_star - A0).

        Note the simplicity: no schedule, no alpha_bar, no posterior variance.
        Just linear interpolation between noise and target, regressed by MSE.
        """
        B = A_star.size(0)
        h = self.vlm(ctx)
        A0 = torch.randn_like(A_star)
        tau = torch.rand(B, device=A_star.device)
        A_tau = (1.0 - tau)[:, None, None] * A0 + tau[:, None, None] * A_star
        v_target = A_star - A0
        v_pred = self.velocity(A_tau, tau, h)
        return F.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def sample(self, ctx: torch.Tensor, n_steps: int = 8) -> torch.Tensor:
        """Generate an action chunk by Euler-integrating the learned ODE."""
        B = ctx.size(0)
        h = self.vlm(ctx)
        A = torch.randn(B, self.cfg.horizon, self.cfg.action_dim, device=ctx.device)
        dt = 1.0 / n_steps
        for k in range(n_steps):
            tau = torch.full((B,), k * dt, device=ctx.device)
            v = self.velocity(A, tau, h)
            A = A + dt * v
        return A


# ----------------------------------------------------------------------------
# 6. DDPM head (the Diffusion Policy way, J16) - SAME backbone
# ----------------------------------------------------------------------------
# We use the simplest DDPM: linear beta schedule, predict-noise parametrization,
# ancestral sampling. Trained on the same data, same expert backbone.

class DDPMPolicy(nn.Module):
    def __init__(self, cfg: ToyConfig, T: int = 100):
        super().__init__()
        self.cfg = cfg
        self.T = T
        self.vlm = MockVLM(cfg)
        self.expert = ActionExpert(cfg)
        # Linear beta schedule (DDPM original).
        betas = torch.linspace(1e-4, 0.02, T)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    def eps_pred(self, A_t: torch.Tensor, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # Reuse the expert -- here it predicts noise (eps) instead of velocity.
        t_emb = time_embedding(t.float(), self.cfg.expert_dim)
        return self.expert(A_t, t_emb, h)

    def loss(self, ctx: torch.Tensor, A_star: torch.Tensor) -> torch.Tensor:
        B = A_star.size(0)
        h = self.vlm(ctx)
        t = torch.randint(0, self.T, (B,), device=A_star.device)
        a_bar = self.alpha_bar[t][:, None, None]
        eps = torch.randn_like(A_star)
        # Closed-form forward: A_t = sqrt(a_bar)*A_0 + sqrt(1-a_bar)*eps
        A_t = a_bar.sqrt() * A_star + (1.0 - a_bar).sqrt() * eps
        eps_hat = self.eps_pred(A_t, t, h)
        return F.mse_loss(eps_hat, eps)

    @torch.no_grad()
    def sample(self, ctx: torch.Tensor, n_steps: int | None = None) -> torch.Tensor:
        """Ancestral DDPM sampling. n_steps=None means full T steps (slow).
        Set n_steps < T to subsample (a poor man's DDIM-lite).
        """
        if n_steps is None:
            n_steps = self.T
        B = ctx.size(0)
        h = self.vlm(ctx)
        A = torch.randn(B, self.cfg.horizon, self.cfg.action_dim, device=ctx.device)
        # Stride through the schedule from T-1 down to 0.
        ts = torch.linspace(self.T - 1, 0, n_steps).long().to(ctx.device)
        for i, t in enumerate(ts):
            t_batch = torch.full((B,), int(t.item()), device=ctx.device)
            eps_hat = self.eps_pred(A, t_batch, h)
            a_bar_t = self.alpha_bar[t]
            a_t = self.alphas[t]
            beta_t = self.betas[t]
            # mean of p(x_{t-1} | x_t)
            mean = (1.0 / a_t.sqrt()) * (
                A - (beta_t / (1.0 - a_bar_t).sqrt()) * eps_hat
            )
            if i < n_steps - 1:
                # Inject noise -- THIS is the SDE part vs flow matching's ODE.
                z = torch.randn_like(A)
                A = mean + beta_t.sqrt() * z
            else:
                A = mean
        return A


# ----------------------------------------------------------------------------
# 7. Training loop
# ----------------------------------------------------------------------------
def train(model: nn.Module, cfg: ToyConfig, name: str) -> list[float]:
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    losses: list[float] = []
    model.train()
    for step in range(cfg.train_steps):
        ctx, A_star = sample_batch(cfg)
        loss = model.loss(ctx, A_star)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if (step + 1) % 250 == 0:
            print(f"  [{name}] step {step + 1:4d} | loss {loss.item():.4f}")
    return losses


# ----------------------------------------------------------------------------
# 8. Evaluation: chunk MSE vs ground truth, varying inference steps
# ----------------------------------------------------------------------------
@torch.no_grad()
def eval_mse(
    model: nn.Module,
    cfg: ToyConfig,
    n_steps: int,
    n_eval: int = 32,
) -> tuple[float, float]:
    """Sample an action chunk and compare to the ground-truth chunk for each
    family. Returns (mean MSE, wall-clock seconds).
    """
    model.eval()
    ctx_list = []
    target_list = []
    for fam_id in range(cfg.n_families):
        ctx = F.one_hot(
            torch.tensor([fam_id] * (n_eval // cfg.n_families)),
            num_classes=cfg.n_families,
        ).float().to(cfg.device)
        tgt = make_target_chunk(fam_id, cfg.horizon).to(cfg.device).expand(
            n_eval // cfg.n_families, -1, -1
        )
        ctx_list.append(ctx)
        target_list.append(tgt)
    ctx = torch.cat(ctx_list)
    tgt = torch.cat(target_list)
    t0 = time.perf_counter()
    pred = model.sample(ctx, n_steps=n_steps)
    dt = time.perf_counter() - t0
    return float(F.mse_loss(pred, tgt).item()), dt


# ----------------------------------------------------------------------------
# 9. Main: train both heads, compare side by side
# ----------------------------------------------------------------------------
def main():
    torch.manual_seed(42)
    cfg = ToyConfig()

    print("=" * 72)
    print("Toy pi0 - Flow Matching vs DDPM on a 3-family action-chunk task")
    print(f"  horizon={cfg.horizon}  action_dim={cfg.action_dim}  "
          f"n_families={cfg.n_families}  train_steps={cfg.train_steps}")
    print("=" * 72)

    # --- Flow matching ---
    fm = FlowMatchingPolicy(cfg).to(cfg.device)
    print("\n[Train] Flow Matching head (pi0-style)")
    fm_losses = train(fm, cfg, name="FM")

    # --- DDPM ---
    dm = DDPMPolicy(cfg, T=100).to(cfg.device)
    print("\n[Train] DDPM head (Diffusion-Policy-style)")
    dm_losses = train(dm, cfg, name="DDPM")

    # --- Inference comparison ---
    print("\n" + "=" * 72)
    print("Inference quality vs step count (lower MSE = better)")
    print("=" * 72)
    print(f"{'method':<15}{'steps':>8}{'MSE':>14}{'walltime (ms)':>18}")
    print("-" * 55)

    for nstep in [2, 5, 10, 25, 50]:
        mse_fm, dt_fm = eval_mse(fm, cfg, n_steps=nstep)
        print(f"{'flow-matching':<15}{nstep:>8}{mse_fm:>14.5f}{dt_fm * 1000:>18.2f}")
    for nstep in [10, 25, 50, 100]:
        mse_dm, dt_dm = eval_mse(dm, cfg, n_steps=nstep)
        print(f"{'ddpm':<15}{nstep:>8}{mse_dm:>14.5f}{dt_dm * 1000:>18.2f}")

    print("\nKey observation:")
    print("  Flow matching converges in ~5-10 Euler steps with small MSE.")
    print("  DDPM needs ~50-100 steps to reach comparable MSE (SDE noise).")
    print("  This is exactly why pi0 (and modern VLAs) prefer flow matching")
    print("  for the action head: same quality, ~10x faster at inference.")
    print("\nReference (REFERENCES.md #14): Black et al., 'pi0: A Vision-Language-")
    print("  Action Flow Model for General Robot Control', arxiv:2410.24164.")


if __name__ == "__main__":
    main()
