# requires: torch>=2.0, numpy
"""
Solutions consolidees J15 - Diffusion + Flow Matching.

Use the CLI dispatcher:
    python 15-diffusion-flow-matching.py easy
    python 15-diffusion-flow-matching.py medium
    python 15-diffusion-flow-matching.py hard

Each `solve_*` function is independent and can be imported directly.
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# EASY -- forward noising in pure numpy
# =============================================================================
def forward_diffuse(
    x0: np.ndarray,
    t: int,
    T: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward DDPM noising at timestep t with linear beta schedule.

    Returns (x_t, eps) where x_t = sqrt(ab_t)*x0 + sqrt(1-ab_t)*eps.
    """
    if not (0 <= t < T):
        raise ValueError(f"t must be in [0, {T - 1}], got {t}")
    rng = rng if rng is not None else np.random.default_rng()
    betas = np.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas)
    eps = rng.standard_normal(x0.shape).astype(x0.dtype)
    ab_t = alpha_bar[t]
    x_t = np.sqrt(ab_t) * x0 + np.sqrt(1.0 - ab_t) * eps
    return x_t, eps


def solve_easy() -> None:
    """Sanity-check forward_diffuse at the 4 prescribed timesteps."""
    rng = np.random.default_rng(0)
    x0 = np.array([1.0, 0.5], dtype=np.float32)
    print(f"x0 = {x0}")
    for t in [0, 100, 500, 999]:
        # Use the same seed for fair comparison across t -- but each call samples its own eps.
        local_rng = np.random.default_rng(0)
        x_t, eps = forward_diffuse(x0, t, T=1000, rng=local_rng)
        norm = np.linalg.norm(x_t)
        print(f"t={t:4d}  x_t={x_t}  |x_t|={norm:.3f}  eps={eps}")
    # Adversarial probe: t at the boundary
    x_T, _ = forward_diffuse(x0, 999, T=1000, rng=np.random.default_rng(123))
    expected = math.sqrt(2)  # 2D N(0, I) ~ |x| close to sqrt(2) on average
    print(f"\nProbe: |x_999| = {np.linalg.norm(x_T):.3f}  (expected ~ {expected:.3f} +/- 1)")


# =============================================================================
# Shared MLP denoiser used by medium + hard
# =============================================================================
class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, device=t.device).float() / half
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class Denoiser(nn.Module):
    """Plain unconditional denoiser used by the medium exercise."""

    def __init__(self, data_dim: int = 2, hidden: int = 128, t_dim: int = 64):
        super().__init__()
        self.t_emb = SinusoidalTimeEmb(t_dim)
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
        return self.net(torch.cat([x, self.t_emb(t)], dim=-1))


class CosineSchedule:
    def __init__(self, T: int = 200, s: float = 0.008, device: str = "cpu"):
        self.T = T
        steps = torch.linspace(0, T, T + 1, device=device) / T
        f = torch.cos(((steps + s) / (1 + s)) * math.pi / 2) ** 2
        alpha_bar = f / f[0]
        beta = torch.clamp(1 - alpha_bar[1:] / alpha_bar[:-1], 1e-5, 0.999)
        self.beta = beta
        self.alpha = 1.0 - beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.device = device

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        ab = self.alpha_bar[t][:, None]
        return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * eps


def make_two_moons(n: int = 4000, noise: float = 0.05, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    half = n // 2
    theta_up = rng.uniform(0, math.pi, half)
    theta_dn = rng.uniform(0, math.pi, n - half)
    upper = np.stack([np.cos(theta_up), np.sin(theta_up)], axis=1)
    lower = np.stack([1.0 - np.cos(theta_dn), -np.sin(theta_dn) - 0.5], axis=1)
    pts = np.concatenate([upper, lower], axis=0)
    pts = pts + rng.normal(0, noise, size=pts.shape)
    pts = ((pts - pts.mean(0)) / pts.std(0)).astype(np.float32)
    return torch.from_numpy(pts)


def make_ring(n: int = 4000, radius: float = 1.0, noise: float = 0.05, seed: int = 1) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * math.pi, n)
    pts = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    pts = pts + rng.normal(0, noise, size=pts.shape)
    pts = pts.astype(np.float32)
    return torch.from_numpy(pts)


# =============================================================================
# MEDIUM -- DDIM sampler from a trained DDPM
# =============================================================================
def train_ddpm(model: Denoiser, sched: CosineSchedule, data: torch.Tensor, epochs: int = 30, batch: int = 512) -> None:
    device = next(model.parameters()).device
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    n = data.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        loss_acc = 0.0
        nb = 0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            x0 = data[idx]
            B = x0.shape[0]
            t = torch.randint(0, sched.T, (B,), device=device)
            eps = torch.randn_like(x0)
            xt = sched.q_sample(x0, t, eps)
            t_norm = t.float() / sched.T
            eps_pred = model(xt, t_norm)
            loss = F.mse_loss(eps_pred, eps)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_acc += loss.item()
            nb += 1
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"[medium-DDPM] ep={ep:3d}  loss={loss_acc / nb:.4f}")


@torch.no_grad()
def my_ddim_sample(model: Denoiser, sched: CosineSchedule, n_samples: int, n_steps: int, seed: int = 0) -> torch.Tensor:
    """Deterministic DDIM sampler (eta=0) -- the medium exercise.

    Picks `n_steps` timesteps roughly evenly spaced in [0, T-1] and walks them
    in descending order. At each step we predict x0 via the eps prediction,
    then re-noise to the next timestep with the *same* eps (eta=0 = deterministic).
    """
    device = next(model.parameters()).device
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(n_samples, 2, device=device, generator=g)
    ts = torch.linspace(sched.T - 1, 0, n_steps, device=device).long()
    for i in range(n_steps):
        t = ts[i].item()
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
        t_norm = t_batch.float() / sched.T
        eps_pred = model(x, t_norm)
        ab_t = sched.alpha_bar[t]
        x0_pred = (x - torch.sqrt(1.0 - ab_t) * eps_pred) / torch.sqrt(ab_t)
        if i + 1 < n_steps:
            t_next = ts[i + 1].item()
            ab_next = sched.alpha_bar[t_next]
            x = torch.sqrt(ab_next) * x0_pred + torch.sqrt(1.0 - ab_next) * eps_pred
        else:
            x = x0_pred
    return x.cpu()


def in_top_moon(samples: torch.Tensor) -> float:
    """Fraction of samples whose y-coord is positive (heuristic: 'top moon')."""
    return (samples[:, 1] > 0).float().mean().item()


def solve_medium() -> None:
    """Train a DDPM on two-moons, sample with DDIM at 10/50/200 steps."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = make_two_moons(n=4000, seed=0).to(device)
    model = Denoiser().to(device)
    sched = CosineSchedule(T=200, device=device)
    train_ddpm(model, sched, data, epochs=30)
    real_top = in_top_moon(data.cpu())
    real_mean = data.mean(0).cpu().numpy().tolist()
    real_std = data.std(0).cpu().numpy().tolist()
    print(f"\nReal data:  mean={real_mean}  std={real_std}  top-moon-frac={real_top:.3f}")
    for n_steps in [10, 50, 200]:
        samples = my_ddim_sample(model, sched, n_samples=1000, n_steps=n_steps, seed=42)
        print(
            f"DDIM steps={n_steps:3d}  mean={samples.mean(0).numpy().tolist()}  "
            f"std={samples.std(0).numpy().tolist()}  top-moon-frac={in_top_moon(samples):.3f}"
        )
    # Reproducibility probe: same seed -> same samples
    s1 = my_ddim_sample(model, sched, n_samples=64, n_steps=20, seed=7)
    s2 = my_ddim_sample(model, sched, n_samples=64, n_steps=20, seed=7)
    diff = (s1 - s2).abs().max().item()
    print(f"\nDeterminism probe (max abs diff between identical seeded calls): {diff:.6f}")


# =============================================================================
# HARD -- Classifier-Free Guidance with a class condition
# =============================================================================
NUM_CLASSES = 2  # 0 = moons, 1 = ring
NULL_TOKEN = NUM_CLASSES  # index reserved for unconditional


class CondDenoiser(nn.Module):
    """Class-conditioned denoiser. NULL_TOKEN provides the unconditional path."""

    def __init__(self, data_dim: int = 2, hidden: int = 128, t_dim: int = 64, c_dim: int = 64):
        super().__init__()
        self.t_emb = SinusoidalTimeEmb(t_dim)
        self.c_emb = nn.Embedding(NUM_CLASSES + 1, c_dim)
        self.net = nn.Sequential(
            nn.Linear(data_dim + t_dim + c_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, data_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, self.t_emb(t), self.c_emb(c)], dim=-1))


def train_cfg(
    model: CondDenoiser,
    sched: CosineSchedule,
    data: torch.Tensor,
    labels: torch.Tensor,
    epochs: int = 40,
    batch: int = 512,
    p_drop: float = 0.15,
) -> None:
    """Joint conditional/unconditional training: drop the label with prob p_drop."""
    device = next(model.parameters()).device
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    n = data.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        loss_acc = 0.0
        nb = 0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            x0 = data[idx]
            c = labels[idx].clone()
            B = x0.shape[0]
            # Drop condition with probability p_drop -> route to NULL_TOKEN
            drop_mask = torch.rand(B, device=device) < p_drop
            c[drop_mask] = NULL_TOKEN
            t = torch.randint(0, sched.T, (B,), device=device)
            eps = torch.randn_like(x0)
            xt = sched.q_sample(x0, t, eps)
            t_norm = t.float() / sched.T
            eps_pred = model(xt, t_norm, c)
            loss = F.mse_loss(eps_pred, eps)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_acc += loss.item()
            nb += 1
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"[hard-CFG] ep={ep:3d}  loss={loss_acc / nb:.4f}")


@torch.no_grad()
def sample_cfg(
    model: CondDenoiser,
    sched: CosineSchedule,
    n: int,
    c_target: int,
    w: float,
    seed: int = 0,
) -> torch.Tensor:
    """Ancestral DDPM sampler with classifier-free guidance.

    eps_guided = (1 + w) * eps_cond - w * eps_uncond
    """
    device = next(model.parameters()).device
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(n, 2, device=device, generator=g)
    c_cond = torch.full((n,), c_target, device=device, dtype=torch.long)
    c_null = torch.full((n,), NULL_TOKEN, device=device, dtype=torch.long)
    T = sched.T
    for t in reversed(range(T)):
        t_batch = torch.full((n,), t, device=device, dtype=torch.long)
        t_norm = t_batch.float() / T
        eps_cond = model(x, t_norm, c_cond)
        eps_uncond = model(x, t_norm, c_null)
        eps_guided = (1.0 + w) * eps_cond - w * eps_uncond
        alpha_t = sched.alpha[t]
        ab_t = sched.alpha_bar[t]
        beta_t = sched.beta[t]
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1.0 - ab_t)) * eps_guided
        )
        if t > 0:
            z = torch.randn(x.shape, device=device, generator=g)
            x = mean + torch.sqrt(beta_t) * z
        else:
            x = mean
    return x.cpu()


def in_ring(samples: torch.Tensor, target_radius: float = 1.0, tol: float = 0.25) -> float:
    """Heuristic geometric test: fraction of samples within a thin annulus."""
    r = samples.norm(dim=1)
    return ((r > target_radius - tol) & (r < target_radius + tol)).float().mean().item()


def in_moons(samples: torch.Tensor, tol: float = 1.5) -> float:
    """Heuristic: moons are roughly inside a disk of radius `tol` and *not* clustered on a circle.

    We use a complementary signal to in_ring: high spread AND not a thin annulus.
    """
    r = samples.norm(dim=1)
    inside_disk = (r < tol).float().mean().item()
    return inside_disk


def total_variance(samples: torch.Tensor) -> float:
    return samples.var(dim=0).sum().item()


def solve_hard() -> None:
    """Train CFG on (moons, ring) and report precision/diversity vs guidance scale w."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    moons = make_two_moons(n=4000, seed=0)
    ring = make_ring(n=4000, radius=1.0, seed=1)
    # Normalize to similar scale
    moons = (moons - moons.mean(0)) / moons.std(0)
    ring = (ring - ring.mean(0)) / ring.std(0).clamp(min=1e-3)
    data = torch.cat([moons, ring], dim=0).to(device)
    labels = torch.cat(
        [
            torch.zeros(moons.shape[0], dtype=torch.long),
            torch.ones(ring.shape[0], dtype=torch.long),
        ]
    ).to(device)
    model = CondDenoiser().to(device)
    sched = CosineSchedule(T=200, device=device)
    train_cfg(model, sched, data, labels, epochs=40, p_drop=0.15)

    print("\n--- CFG sweep ---")
    print(f"{'class':>8s} {'w':>5s} {'precision':>10s} {'variance':>10s}")
    for c_target, name in [(0, "moons"), (1, "ring")]:
        for w in [0.0, 1.0, 3.0, 7.0]:
            samples = sample_cfg(model, sched, n=500, c_target=c_target, w=w, seed=42)
            if c_target == 1:
                precision = in_ring(samples, target_radius=1.0, tol=0.4)
            else:
                # Two-moons should NOT be a tight ring -> use complementary metric
                precision = 1.0 - in_ring(samples, target_radius=1.0, tol=0.15)
            var = total_variance(samples)
            print(f"{name:>8s} {w:>5.1f} {precision:>10.3f} {var:>10.3f}")


# =============================================================================
# CLI dispatcher
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Solutions J15 (easy / medium / hard)")
    parser.add_argument("level", choices=["easy", "medium", "hard", "all"])
    args = parser.parse_args()
    if args.level in {"easy", "all"}:
        print("\n=========================  EASY  =========================")
        solve_easy()
    if args.level in {"medium", "all"}:
        print("\n=========================  MEDIUM  =======================")
        solve_medium()
    if args.level in {"hard", "all"}:
        print("\n=========================  HARD  =========================")
        solve_hard()


if __name__ == "__main__":
    sys.exit(main() or 0)
