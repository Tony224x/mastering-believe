"""
J16 — Diffusion Policy (Chi et al., RSS 2023) — pedagogical mini-implementation.

This file mirrors the three core building blocks of `real-stanford/diffusion_policy`:
    1. Visual + state conditioning (ResNet18-style encoder)
    2. Conditional UNet1D denoiser with FiLM modulation + sinusoidal timestep embedding
    3. DDPM forward/reverse process (squared-cosine schedule)
on a TOY 2D push-T-like multimodal action distribution. No MuJoCo, no real dataset:
the goal is to show every brick assembled in <300 lines, runnable on CPU in <30 s.

Reference (J16 theory): REFERENCES.md #19 (Chi 2023) and #23 (MIT 6.S184 notes).

# requires: torch>=2.1, numpy>=1.24
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Toy multimodal dataset
# ---------------------------------------------------------------------------
# We build a dataset where, for a given observation `o` (a scalar in [-1,1]),
# the expert action SEQUENCE has TWO valid strategies. A naive MSE BC will
# average them and fail; diffusion policy will recover both modes.
def make_toy_dataset(n_samples: int = 4096, horizon: int = 8, seed: int = 0):
    """Return tensors (obs[N,1], actions[N, horizon, 2]) with bimodal action chunks."""
    rng = np.random.default_rng(seed)
    obs = rng.uniform(-1.0, 1.0, size=(n_samples, 1)).astype(np.float32)
    actions = np.zeros((n_samples, horizon, 2), dtype=np.float32)
    # For each sample, pick strategy A (curve up-then-right) or B (right-then-up).
    # Both are valid given `obs`. A naive regressor will collapse to their mean.
    t = np.linspace(0.0, 1.0, horizon, dtype=np.float32)
    for i in range(n_samples):
        choose_A = rng.random() < 0.5
        amp = 0.3 + 0.2 * obs[i, 0]  # observation modulates amplitude
        if choose_A:
            actions[i, :, 0] = amp * np.sin(np.pi * t)        # x: arch
            actions[i, :, 1] = amp * t                        # y: monotonic
        else:
            actions[i, :, 0] = amp * t                        # x: monotonic
            actions[i, :, 1] = amp * np.sin(np.pi * t)        # y: arch
        actions[i] += rng.normal(0.0, 0.01, size=actions[i].shape).astype(np.float32)
    return torch.from_numpy(obs), torch.from_numpy(actions)


# ---------------------------------------------------------------------------
# 2. Conditioning encoder — pedagogical stand-in for ResNet18 + state MLP
# ---------------------------------------------------------------------------
# In the real repo this is `model/vision/model_getter.py` (ResNet18 with
# BatchNorm replaced by GroupNorm). For our toy scalar `obs` we use a small MLP.
class ObsEncoder(nn.Module):
    def __init__(self, obs_dim: int = 1, cond_dim: int = 64):
        super().__init__()
        # Tiny MLP — real DiffusionUnetImagePolicy plugs a (frozen-init) ResNet18
        # here. The output dim `cond_dim` is what is fed into FiLM blocks below.
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Mish(),
            nn.Linear(64, cond_dim),
            nn.Mish(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ---------------------------------------------------------------------------
# 3. Sinusoidal timestep embedding — same trick as in DDPM/Stable Diffusion
# ---------------------------------------------------------------------------
class SinusoidalTimeEmb(nn.Module):
    """Maps an integer timestep k -> dense vector. Identical to Ho 2020."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        # k: [B] integer tensor of diffusion timesteps in {0, ..., K-1}
        half = self.dim // 2
        # Frequencies geometrically spaced (log-linear) — paper formula.
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=k.device, dtype=torch.float32) / max(half - 1, 1)
        )
        args = k.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ---------------------------------------------------------------------------
# 4. FiLM-conditioned 1D residual block — heart of the UNet1D denoiser
# ---------------------------------------------------------------------------
class FiLMBlock1D(nn.Module):
    """Conv1D + GroupNorm + Mish, with FiLM(gamma, beta) injection of cond/time."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, n_groups: int = 8):
        super().__init__()
        # GroupNorm (NOT BatchNorm) — Chi 2023 explicitly warns BN breaks with EMA.
        self.norm1 = nn.GroupNorm(min(n_groups, in_ch), in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(n_groups, out_ch), out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        # FiLM linear projects (cond + time) into per-channel (gamma, beta).
        self.film = nn.Linear(cond_dim, 2 * out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]  cond: [B, cond_dim]
        residual = self.skip(x)
        h = self.conv1(F.mish(self.norm1(x)))
        gamma_beta = self.film(cond)  # [B, 2*out_ch]
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        # FiLM = feature-wise affine modulation, broadcast over time axis.
        h = h * (1.0 + gamma[:, :, None]) + beta[:, :, None]
        h = self.conv2(F.mish(self.norm2(h)))
        return h + residual


# ---------------------------------------------------------------------------
# 5. ConditionalUNet1D — the actual denoiser ε_θ(a^k, k, c)
# ---------------------------------------------------------------------------
class ConditionalUNet1D(nn.Module):
    """Mini UNet1D over the action sequence axis, FiLM-conditioned by (obs, time)."""

    def __init__(self, action_dim: int, cond_dim: int, time_dim: int = 64,
                 hidden: tuple[int, ...] = (64, 128, 256)):
        super().__init__()
        self.time_emb = SinusoidalTimeEmb(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4), nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )
        # We merge (obs cond, time emb) into a single conditioning vector.
        merged_cond = cond_dim + time_dim
        # Down path
        self.down1 = FiLMBlock1D(action_dim, hidden[0], merged_cond)
        self.down2 = FiLMBlock1D(hidden[0], hidden[1], merged_cond)
        # Bottleneck
        self.mid = FiLMBlock1D(hidden[1], hidden[2], merged_cond)
        # Up path with skip connections (UNet pattern)
        self.up2 = FiLMBlock1D(hidden[2] + hidden[1], hidden[1], merged_cond)
        self.up1 = FiLMBlock1D(hidden[1] + hidden[0], hidden[0], merged_cond)
        self.out = nn.Conv1d(hidden[0], action_dim, kernel_size=1)

    def forward(self, a_noisy: torch.Tensor, k: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # a_noisy: [B, T, A]  ->  permute to [B, A, T] for Conv1d
        x = a_noisy.permute(0, 2, 1)
        t_emb = self.time_mlp(self.time_emb(k))   # [B, time_dim]
        cond = torch.cat([c, t_emb], dim=-1)       # [B, merged_cond]
        h1 = self.down1(x, cond)
        h2 = self.down2(h1, cond)
        m = self.mid(h2, cond)
        u2 = self.up2(torch.cat([m, h2], dim=1), cond)
        u1 = self.up1(torch.cat([u2, h1], dim=1), cond)
        eps_pred = self.out(u1)                     # [B, A, T]
        return eps_pred.permute(0, 2, 1)            # back to [B, T, A]


# ---------------------------------------------------------------------------
# 6. DDPM scheduler with squared-cosine schedule (Chi 2023 default)
# ---------------------------------------------------------------------------
@dataclass
class DDPMSchedule:
    K: int                       # number of diffusion steps
    betas: torch.Tensor          # [K]
    alphas: torch.Tensor         # [K]
    alphas_cumprod: torch.Tensor  # [K]


def squared_cosine_schedule(K: int, s: float = 0.008) -> DDPMSchedule:
    """Nichol & Dhariwal 2021 cosine schedule (also default in Chi 2023 repo)."""
    steps = torch.arange(K + 1, dtype=torch.float32)
    f = torch.cos(((steps / K) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = f / f[0]
    betas = torch.clamp(1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1], 1e-4, 0.999)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return DDPMSchedule(K=K, betas=betas, alphas=alphas, alphas_cumprod=alphas_cumprod)


def q_sample(a_clean: torch.Tensor, k: torch.Tensor, sched: DDPMSchedule) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward noising: a^k = sqrt(abar)·a^0 + sqrt(1-abar)·eps. Returns (a_noisy, eps)."""
    abar = sched.alphas_cumprod[k][:, None, None]              # [B,1,1]
    eps = torch.randn_like(a_clean)
    a_noisy = abar.sqrt() * a_clean + (1.0 - abar).sqrt() * eps
    return a_noisy, eps


@torch.no_grad()
def ddpm_sample(model: ConditionalUNet1D, encoder: ObsEncoder, obs: torch.Tensor,
                horizon: int, action_dim: int, sched: DDPMSchedule) -> torch.Tensor:
    """Reverse DDPM: start from N(0,I), iterate K denoising steps."""
    device = obs.device
    a = torch.randn(obs.shape[0], horizon, action_dim, device=device)
    c = encoder(obs)
    for k in reversed(range(sched.K)):
        k_batch = torch.full((obs.shape[0],), k, device=device, dtype=torch.long)
        eps_pred = model(a, k_batch, c)
        alpha_k = sched.alphas[k]
        abar_k = sched.alphas_cumprod[k]
        # Posterior mean (Ho 2020 eq. 11) — conditioned on predicted eps.
        coef = (1.0 - alpha_k) / (1.0 - abar_k).sqrt()
        mean = (a - coef * eps_pred) / alpha_k.sqrt()
        if k > 0:
            noise = torch.randn_like(a)
            sigma = sched.betas[k].sqrt()
            a = mean + sigma * noise
        else:
            a = mean
    return a


# ---------------------------------------------------------------------------
# 7. Training loop — MSE on noise (Ho 2020 simplified loss)
# ---------------------------------------------------------------------------
def train(epochs: int = 30, batch_size: int = 256, lr: float = 1e-3,
          K: int = 100, horizon: int = 8, action_dim: int = 2,
          device: str = "cpu", seed: int = 0) -> dict:
    torch.manual_seed(seed)
    obs, actions = make_toy_dataset(n_samples=4096, horizon=horizon, seed=seed)
    obs, actions = obs.to(device), actions.to(device)
    encoder = ObsEncoder(obs_dim=1, cond_dim=64).to(device)
    model = ConditionalUNet1D(action_dim=action_dim, cond_dim=64).to(device)
    sched_ = squared_cosine_schedule(K)
    sched = DDPMSchedule(
        K=sched_.K,
        betas=sched_.betas.to(device),
        alphas=sched_.alphas.to(device),
        alphas_cumprod=sched_.alphas_cumprod.to(device),
    )
    opt = torch.optim.AdamW(list(model.parameters()) + list(encoder.parameters()), lr=lr)
    n = obs.shape[0]
    losses: list[float] = []
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        ep_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            o_b, a_b = obs[idx], actions[idx]
            k = torch.randint(0, sched.K, (o_b.shape[0],), device=device)
            a_noisy, eps = q_sample(a_b, k, sched)
            c = encoder(o_b)
            eps_pred = model(a_noisy, k, c)
            loss = F.mse_loss(eps_pred, eps)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * o_b.shape[0]
        ep_loss /= n
        losses.append(ep_loss)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"epoch {ep:03d} | loss {ep_loss:.5f}")
    return {"encoder": encoder, "model": model, "sched": sched, "losses": losses}


# ---------------------------------------------------------------------------
# 8. Entry point — train, then sample to *show* the multimodality is recovered.
# ---------------------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    out = train(epochs=20, device=device)
    # At test time, sample 64 action chunks for the SAME observation.
    obs_test = torch.zeros(64, 1, device=device)  # neutral observation
    samples = ddpm_sample(out["model"], out["encoder"], obs_test,
                          horizon=8, action_dim=2, sched=out["sched"])
    # Bucket samples into "strategy A" (x peaks early) vs "B" (y peaks early).
    x_peak_t = samples[:, :, 0].argmax(dim=1).float().mean().item()
    y_peak_t = samples[:, :, 1].argmax(dim=1).float().mean().item()
    var_x = samples[:, :, 0].argmax(dim=1).float().var().item()
    print(f"\nSampled 64 chunks for o=0:")
    print(f"  mean argmax over time, x-axis: {x_peak_t:.2f}  (mode A peaks ~3.5, B near end)")
    print(f"  mean argmax over time, y-axis: {y_peak_t:.2f}  (mode A near end, B peaks ~3.5)")
    print(f"  var of x-axis argmax across samples: {var_x:.2f}  (high var = bimodality preserved)")
    # A naive BC would collapse argmax variance toward 0; diffusion keeps it >0.


if __name__ == "__main__":
    main()
