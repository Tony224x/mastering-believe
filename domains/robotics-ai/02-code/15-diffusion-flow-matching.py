# requires: torch>=2.0, numpy, matplotlib
"""
J15 - Diffusion + Flow Matching unified, pedagogical implementation.

We train a tiny DDPM on a 2D toy dataset (two-moons) to see the full machinery
in ~30 seconds on CPU. The same code is reused to compare:
    1. DDPM (stochastic SDE-style sampler)
    2. DDIM (deterministic, fewer steps)
    3. Flow Matching (linear interpolant + ODE Euler sampler)

This is the minimal substrate needed to understand Diffusion Policy (J16) and
the action heads of pi0 / GR00T N1 (J21-J22). Same math, different data shape.

Run:
    python 15-diffusion-flow-matching.py            # trains DDPM + samples
    python 15-diffusion-flow-matching.py --mode fm  # trains Flow Matching
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# 1) Toy dataset: two-moons in 2D (multimodal -> a great DDPM stress test)
# -----------------------------------------------------------------------------
def make_two_moons(n: int = 8000, noise: float = 0.05, seed: int = 0) -> torch.Tensor:
    """Generate n points on the two-moons distribution. No sklearn dependency."""
    rng = np.random.default_rng(seed)
    half = n // 2
    # Upper moon (y >= 0) and lower moon (y <= 0) interleaved
    theta_up = rng.uniform(0, math.pi, half)
    theta_dn = rng.uniform(0, math.pi, n - half)
    upper = np.stack([np.cos(theta_up), np.sin(theta_up)], axis=1)
    lower = np.stack([1.0 - np.cos(theta_dn), -np.sin(theta_dn) - 0.5], axis=1)
    pts = np.concatenate([upper, lower], axis=0)
    pts = pts + rng.normal(0, noise, size=pts.shape)
    pts = pts.astype(np.float32)
    # Center & scale so the distribution lives roughly in the unit ball
    pts = (pts - pts.mean(0)) / pts.std(0)
    return torch.from_numpy(pts)


# -----------------------------------------------------------------------------
# 2) Tiny denoiser MLP -- conditioned on a sinusoidal time embedding
#    Same network is reused for DDPM (predicts eps) and Flow Matching (predicts v)
# -----------------------------------------------------------------------------
class SinusoidalTimeEmb(nn.Module):
    """Standard transformer-style sinusoidal embedding for the timestep t."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) float in [0, 1] (we rescale DDPM integer t -> [0, 1] before)
        half = self.dim // 2
        # log-spaced frequencies, classic transformer recipe
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, device=t.device).float() / half
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class Denoiser(nn.Module):
    def __init__(self, data_dim: int = 2, hidden: int = 128, t_dim: int = 64):
        super().__init__()
        self.t_emb = SinusoidalTimeEmb(t_dim)
        # We concat [x, t_emb] then run a 3-layer MLP. Tiny but enough for 2D toys.
        self.net = nn.Sequential(
            nn.Linear(data_dim + t_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, data_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x, self.t_emb(t)], dim=-1)
        return self.net(h)


# -----------------------------------------------------------------------------
# 3) DDPM scheduler -- cosine schedule (Nichol & Dhariwal 2021)
#    Predict noise eps; loss = MSE(eps_pred, eps).
# -----------------------------------------------------------------------------
class CosineSchedule:
    """Holds alphas/betas for T discrete steps. Pure data, no learnable params."""

    def __init__(self, T: int = 200, s: float = 0.008, device: str = "cpu"):
        self.T = T
        # alpha_bar_t = cos((t/T + s)/(1+s) * pi/2)^2  -- the formula from the theory file
        steps = torch.linspace(0, T, T + 1, device=device) / T
        f = torch.cos(((steps + s) / (1 + s)) * math.pi / 2) ** 2
        alpha_bar = f / f[0]
        # Numerical safety: clamp betas to [1e-5, 0.999]
        beta = torch.clamp(1 - alpha_bar[1:] / alpha_bar[:-1], 1e-5, 0.999)
        self.beta = beta  # (T,)
        self.alpha = 1.0 - beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)  # (T,)
        self.device = device

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Forward noising: x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*eps"""
        ab = self.alpha_bar[t][:, None]
        return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * eps


def train_ddpm(epochs: int = 40, batch: int = 512, T: int = 200) -> tuple[Denoiser, CosineSchedule]:
    """Train a DDPM on two-moons. Returns (model, schedule)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = make_two_moons(n=8000).to(device)
    model = Denoiser().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = CosineSchedule(T=T, device=device)

    n = data.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            x0 = data[idx]
            B = x0.shape[0]
            # Sample timestep uniformly in {0, ..., T-1}
            t = torch.randint(0, T, (B,), device=device)
            eps = torch.randn_like(x0)
            xt = sched.q_sample(x0, t, eps)
            # Network sees normalized time in [0, 1] -- matches sinusoidal emb scale
            t_norm = t.float() / T
            eps_pred = model(xt, t_norm)
            loss = F.mse_loss(eps_pred, eps)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"[DDPM] epoch {ep:3d}  loss={epoch_loss / n_batches:.4f}")
    return model, sched


@torch.no_grad()
def sample_ddpm(model: Denoiser, sched: CosineSchedule, n: int = 1000) -> torch.Tensor:
    """Ancestral sampler: T denoising steps with stochastic noise z."""
    device = next(model.parameters()).device
    x = torch.randn(n, 2, device=device)  # x_T ~ N(0, I)
    T = sched.T
    for t in reversed(range(T)):
        t_batch = torch.full((n,), t, device=device, dtype=torch.long)
        t_norm = t_batch.float() / T
        eps_pred = model(x, t_norm)
        alpha_t = sched.alpha[t]
        alpha_bar_t = sched.alpha_bar[t]
        beta_t = sched.beta[t]
        # Mean of p(x_{t-1} | x_t)
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_pred
        )
        if t > 0:
            z = torch.randn_like(x)
            sigma = torch.sqrt(beta_t)
            x = mean + sigma * z
        else:
            x = mean
    return x.cpu()


@torch.no_grad()
def sample_ddim(model: Denoiser, sched: CosineSchedule, n: int = 1000, steps: int = 50) -> torch.Tensor:
    """DDIM deterministic sampler: subset of timesteps, eta=0 (no stochastic noise)."""
    device = next(model.parameters()).device
    T = sched.T
    # Pick `steps` timesteps roughly evenly spaced, in descending order
    ts = torch.linspace(T - 1, 0, steps, device=device).long()
    x = torch.randn(n, 2, device=device)
    for i in range(steps):
        t = ts[i].item()
        t_batch = torch.full((n,), t, device=device, dtype=torch.long)
        t_norm = t_batch.float() / T
        eps_pred = model(x, t_norm)
        alpha_bar_t = sched.alpha_bar[t]
        # Predict x_0 from x_t and eps
        x0_pred = (x - torch.sqrt(1.0 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
        if i + 1 < steps:
            t_next = ts[i + 1].item()
            alpha_bar_next = sched.alpha_bar[t_next]
            # DDIM eta=0 update: x_{t-1} = sqrt(ab_next)*x0 + sqrt(1-ab_next)*eps
            x = torch.sqrt(alpha_bar_next) * x0_pred + torch.sqrt(1.0 - alpha_bar_next) * eps_pred
        else:
            x = x0_pred
    return x.cpu()


# -----------------------------------------------------------------------------
# 4) Flow Matching (linear interpolant)
#    Path:    x_t = (1 - t)*x_1 + t*x_0,    t in [0, 1], x_1 ~ N(0, I)
#    Target:  v(x_t, t) = x_0 - x_1
#    Sampler: ODE Euler integration from t=0 to t=1
# -----------------------------------------------------------------------------
def train_flow_matching(epochs: int = 40, batch: int = 512) -> Denoiser:
    """Train a Flow Matching model. Same denoiser net, different target."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = make_two_moons(n=8000).to(device)
    model = Denoiser().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    n = data.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            x0 = data[idx]
            B = x0.shape[0]
            x1 = torch.randn_like(x0)  # noise endpoint
            t = torch.rand(B, device=device)  # uniform time in [0, 1]
            xt = (1.0 - t)[:, None] * x1 + t[:, None] * x0  # interpolant
            v_target = x0 - x1  # constant velocity along linear path
            v_pred = model(xt, t)
            loss = F.mse_loss(v_pred, v_target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"[FM] epoch {ep:3d}  loss={epoch_loss / n_batches:.4f}")
    return model


@torch.no_grad()
def sample_flow_matching(model: Denoiser, n: int = 1000, steps: int = 50) -> torch.Tensor:
    """Euler-integrate the learned ODE from t=0 (noise) to t=1 (data)."""
    device = next(model.parameters()).device
    x = torch.randn(n, 2, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((n,), i * dt, device=device)
        v = model(x, t)
        x = x + dt * v  # explicit Euler step
    return x.cpu()


# -----------------------------------------------------------------------------
# 5) Eval helpers
# -----------------------------------------------------------------------------
def two_moments_summary(samples: torch.Tensor) -> dict:
    """Quick numerical sanity check: mean and std of samples."""
    return {
        "mean": samples.mean(0).numpy().tolist(),
        "std": samples.std(0).numpy().tolist(),
    }


def save_scatter(samples: torch.Tensor, real: torch.Tensor, path: str) -> None:
    """Save a side-by-side scatter plot of real vs generated. Optional matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001 - matplotlib optional at runtime
        print(f"matplotlib not available, skipping {path}")
        return
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].scatter(real[:, 0], real[:, 1], s=2, alpha=0.5)
    axes[0].set_title("Real two-moons")
    axes[0].set_aspect("equal")
    axes[1].scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.5, c="orange")
    axes[1].set_title("Generated")
    axes[1].set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"saved {path}")


# -----------------------------------------------------------------------------
# 6) Main / CLI
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ddpm", "fm", "both"], default="ddpm")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--T", type=int, default=200, help="DDPM diffusion steps")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--save-plots", action="store_true")
    args = parser.parse_args()

    real = make_two_moons(n=2000, seed=42)

    if args.mode in {"ddpm", "both"}:
        print("=" * 60)
        print("Training DDPM with cosine schedule...")
        model, sched = train_ddpm(epochs=args.epochs, T=args.T)
        ddpm_samples = sample_ddpm(model, sched, n=args.n_samples)
        ddim_samples = sample_ddim(model, sched, n=args.n_samples, steps=50)
        print("DDPM ancestral samples:", two_moments_summary(ddpm_samples))
        print("DDIM 50-step samples  :", two_moments_summary(ddim_samples))
        if args.save_plots:
            save_scatter(ddpm_samples, real, "ddpm_samples.png")
            save_scatter(ddim_samples, real, "ddim_samples.png")

    if args.mode in {"fm", "both"}:
        print("=" * 60)
        print("Training Flow Matching (linear interpolant)...")
        model_fm = train_flow_matching(epochs=args.epochs)
        fm_samples = sample_flow_matching(model_fm, n=args.n_samples, steps=50)
        print("Flow Matching ODE samples:", two_moments_summary(fm_samples))
        if args.save_plots:
            save_scatter(fm_samples, real, "fm_samples.png")


if __name__ == "__main__":
    main()
