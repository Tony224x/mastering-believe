# requires: torch numpy
"""
J21 - Solutions consolidees pour les 3 exercices.

EASY    : flow_matching_step + euler_integrate (numpy/torch).
MEDIUM  : multi-embodiment padding + masked MSE loss policy.
HARD    : mini pi0 (FM head + DDPM head) avec masking multi-embodiment,
          benchmark MSE-vs-steps + walltime, commentaire automatique.

Usage:
    python domains/robotics-ai/03-exercises/solutions/21-pi0-physical-intelligence.py [easy|medium|hard|all]

Source: REFERENCES.md #14, Black et al., pi0, 2024.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# EASY -- flow_matching_step + euler_integrate
# ============================================================================
def flow_matching_step(
    A_tau: torch.Tensor,
    tau: torch.Tensor,
    A_star: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Return the velocity-target (A_star - A_0) given an interpolated point.

    Recall: A_tau = (1 - tau) * A_0 + tau * A_star
    => A_0 = (A_tau - tau * A_star) / (1 - tau)   (clip 1-tau to avoid /0)
    => velocity-target = A_star - A_0
    """
    one_minus_tau = (1.0 - tau).clamp_min(eps)
    # broadcast tau to match (B, H, d)
    tau_b = tau.view(-1, *([1] * (A_tau.dim() - 1)))
    omt_b = one_minus_tau.view(-1, *([1] * (A_tau.dim() - 1)))
    A_0 = (A_tau - tau_b * A_star) / omt_b
    return A_star - A_0


def euler_integrate(velocity_fn, A_init: torch.Tensor, n_steps: int) -> torch.Tensor:
    """Solve dA/d_tau = velocity_fn(A, tau) via N Euler steps from tau=0 to 1."""
    A = A_init.clone()
    dt = 1.0 / n_steps
    B = A.size(0)
    for k in range(n_steps):
        tau = torch.full((B,), k * dt, device=A.device)
        v = velocity_fn(A, tau)
        A = A + dt * v
    return A


def run_easy() -> None:
    print("=" * 60)
    print("EASY: flow_matching_step + euler_integrate")
    print("=" * 60)
    B, H, d = 4, 16, 2
    torch.manual_seed(0)

    # 1. Sanity-check flow_matching_step on a known case.
    A_star = torch.ones(B, H, d)
    A_0 = torch.zeros(B, H, d)
    tau = torch.tensor([0.3] * B)
    A_tau = (1 - 0.3) * A_0 + 0.3 * A_star  # = 0.3
    v_target = flow_matching_step(A_tau, tau, A_star)
    expected = A_star - A_0  # = 1.0 everywhere
    print(f"  velocity-target close to 1.0?  err={(v_target - expected).abs().max().item():.2e}")
    assert (v_target - expected).abs().max().item() < 1e-5

    # 2. Euler integration: trivial constant-velocity field that pulls A toward target.
    target = torch.ones(B, H, d)
    A_init = torch.randn(B, H, d)
    velocity_fn = lambda A, tau: target - A  # noqa: E731

    for n_steps in [2, 5, 10, 25]:
        A_final = euler_integrate(velocity_fn, A_init.clone(), n_steps)
        mse = F.mse_loss(A_final, target).item()
        print(f"  Euler n_steps={n_steps:>3}  ->  MSE to target = {mse:.5f}")

    print("  Observation: more Euler steps => MSE drops -- standard ODE accuracy.")
    print("  Why no noise injection? FM is an ODE: velocity field is fully")
    print("  deterministic, integrating it is integration. DDPM is an SDE:")
    print("  the reverse process is *itself* stochastic (z ~ N(0,I) per step).\n")


# ============================================================================
# MEDIUM -- multi-embodiment padding + masked MSE policy
# ============================================================================
EMB_DIMS = {0: 2, 1: 4, 2: 7}     # embodiment_id -> action dim
DIM_MAX = max(EMB_DIMS.values())  # 7


def make_targets_and_masks(emb_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """For each embodiment id in the batch, produce a (target, mask) padded to DIM_MAX.

    Target = a deterministic function of (emb_id, dim_index) -> sin pattern.
    Mask   = 1 on valid dims, 0 on padded dims.
    """
    B = emb_ids.size(0)
    targets = torch.zeros(B, DIM_MAX)
    masks = torch.zeros(B, DIM_MAX)
    for i, eid in enumerate(emb_ids.tolist()):
        d = EMB_DIMS[int(eid)]
        # toy target: sin((eid + 1) * (k + 1) * 0.5) for valid dims
        for k in range(d):
            targets[i, k] = math.sin((eid + 1) * (k + 1) * 0.5)
        masks[i, :d] = 1.0
    return targets, masks


class MaskedPolicy(nn.Module):
    """Tiny MLP context -> action_padded, trained with masked MSE."""

    def __init__(self, n_emb: int = 3, ctx_dim: int = 8):
        super().__init__()
        self.n_emb = n_emb
        in_dim = ctx_dim + n_emb
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, DIM_MAX),
        )

    def forward(self, ctx: torch.Tensor, emb_ids: torch.Tensor) -> torch.Tensor:
        emb_oh = F.one_hot(emb_ids, num_classes=self.n_emb).float()
        x = torch.cat([ctx, emb_oh], dim=-1)
        return self.net(x)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    sq = (pred - target) ** 2
    return (mask * sq).sum() / mask.sum().clamp_min(1.0)


def run_medium() -> None:
    print("=" * 60)
    print("MEDIUM: multi-embodiment padding + masked MSE")
    print("=" * 60)
    torch.manual_seed(1)
    policy = MaskedPolicy()
    opt = torch.optim.Adam(policy.parameters(), lr=3e-3)

    BATCH = 128
    N_STEPS = 1500

    for step in range(N_STEPS):
        # sample random embodiments
        emb_ids = torch.randint(0, 3, (BATCH,))
        # context: random 8-d vector (no semantic role here; pretend it's vision)
        ctx = torch.randn(BATCH, 8)
        targets, masks = make_targets_and_masks(emb_ids)
        pred = policy(ctx, emb_ids)
        loss = masked_mse(pred, targets, masks)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 250 == 0:
            print(f"  step {step + 1:4d}  masked-loss={loss.item():.4f}")

    # Per-embodiment evaluation
    policy.eval()
    print("\n  Per-embodiment evaluation:")
    print(f"  {'emb_id':<8}{'valid_dim':<12}{'MSE_valid':<12}{'MSE_padded':<12}")
    print("  " + "-" * 44)
    with torch.no_grad():
        for eid in range(3):
            B = 256
            ctx = torch.randn(B, 8)
            emb_ids = torch.full((B,), eid, dtype=torch.long)
            targets, masks = make_targets_and_masks(emb_ids)
            pred = policy(ctx, emb_ids)
            d = EMB_DIMS[eid]
            mse_valid = ((pred[:, :d] - targets[:, :d]) ** 2).mean().item()
            if d < DIM_MAX:
                mse_padded = (pred[:, d:] ** 2).mean().item()
            else:
                mse_padded = float("nan")
            print(f"  {eid:<8}{d:<12}{mse_valid:<12.5f}{mse_padded:<12.5f}")
    print("  Observation: MSE on valid dims converges; padded dims free to vary.")
    print("  Why this works: pi0 pads to DIM_MAX=18 across 7 robots. The mask")
    print("  forces gradients only on the dims that physically exist for that")
    print("  embodiment, so a single net learns 7 different action spaces.\n")


# ============================================================================
# HARD -- mini pi0: FM head + DDPM head, masked, benchmarked
# ============================================================================
@dataclass
class HardConfig:
    n_emb: int = 3
    horizon: int = 12
    dim_max: int = 7
    train_steps: int = 1500
    batch_size: int = 64
    expert_dim: int = 64
    vlm_dim: int = 32
    device: str = "cpu"


def make_chunk_targets(emb_ids: torch.Tensor, H: int) -> tuple[torch.Tensor, torch.Tensor]:
    """For each emb in batch, produce chunk (H, DIM_MAX) and mask (H, DIM_MAX)."""
    B = emb_ids.size(0)
    chunks = torch.zeros(B, H, DIM_MAX)
    masks = torch.zeros(B, H, DIM_MAX)
    t = torch.linspace(0.0, 2.0 * math.pi, H)
    for i, eid in enumerate(emb_ids.tolist()):
        d = EMB_DIMS[int(eid)]
        for k in range(d):
            chunks[i, :, k] = torch.sin((eid + 1) * (k + 1) * 0.5 + t)
        masks[i, :, :d] = 1.0
    return chunks, masks


def time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t.unsqueeze(-1).float() * freqs.unsqueeze(0)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class MockVLM(nn.Module):
    def __init__(self, cfg: HardConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_emb, cfg.vlm_dim), nn.GELU(),
            nn.Linear(cfg.vlm_dim, cfg.vlm_dim),
        )

    def forward(self, x):
        return self.net(x)


class ActionExpert(nn.Module):
    def __init__(self, cfg: HardConfig):
        super().__init__()
        self.cfg = cfg
        flat = cfg.horizon * cfg.dim_max
        self.net = nn.Sequential(
            nn.Linear(flat + cfg.expert_dim + cfg.vlm_dim, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, flat),
        )

    def forward(self, A, t_emb, h):
        B = A.size(0)
        x = torch.cat([A.reshape(B, -1), t_emb, h], dim=-1)
        return self.net(x).reshape(B, self.cfg.horizon, self.cfg.dim_max)


class FMPolicyMasked(nn.Module):
    def __init__(self, cfg: HardConfig):
        super().__init__()
        self.cfg = cfg
        self.vlm = MockVLM(cfg)
        self.expert = ActionExpert(cfg)

    def velocity(self, A_tau, tau, h):
        return self.expert(A_tau, time_embedding(tau, self.cfg.expert_dim), h)

    def loss(self, ctx, A_star, mask):
        B = A_star.size(0)
        h = self.vlm(ctx)
        A0 = torch.randn_like(A_star)
        tau = torch.rand(B, device=A_star.device)
        A_tau = (1 - tau)[:, None, None] * A0 + tau[:, None, None] * A_star
        v_target = A_star - A0
        v_pred = self.velocity(A_tau, tau, h)
        sq = (v_pred - v_target) ** 2
        return (mask * sq).sum() / mask.sum().clamp_min(1.0)

    @torch.no_grad()
    def sample(self, ctx, n_steps=8):
        B = ctx.size(0)
        h = self.vlm(ctx)
        A = torch.randn(B, self.cfg.horizon, self.cfg.dim_max, device=ctx.device)
        dt = 1.0 / n_steps
        for k in range(n_steps):
            tau = torch.full((B,), k * dt, device=ctx.device)
            v = self.velocity(A, tau, h)
            A = A + dt * v
        return A


class DDPMPolicyMasked(nn.Module):
    def __init__(self, cfg: HardConfig, T: int = 100):
        super().__init__()
        self.cfg = cfg
        self.T = T
        self.vlm = MockVLM(cfg)
        self.expert = ActionExpert(cfg)
        betas = torch.linspace(1e-4, 0.02, T)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    def eps_pred(self, A, t, h):
        return self.expert(A, time_embedding(t.float(), self.cfg.expert_dim), h)

    def loss(self, ctx, A_star, mask):
        B = A_star.size(0)
        h = self.vlm(ctx)
        t = torch.randint(0, self.T, (B,), device=A_star.device)
        a_bar = self.alpha_bar[t][:, None, None]
        eps = torch.randn_like(A_star)
        A_t = a_bar.sqrt() * A_star + (1 - a_bar).sqrt() * eps
        eps_hat = self.eps_pred(A_t, t, h)
        sq = (eps_hat - eps) ** 2
        return (mask * sq).sum() / mask.sum().clamp_min(1.0)

    @torch.no_grad()
    def sample(self, ctx, n_steps=None):
        if n_steps is None:
            n_steps = self.T
        B = ctx.size(0)
        h = self.vlm(ctx)
        A = torch.randn(B, self.cfg.horizon, self.cfg.dim_max, device=ctx.device)
        ts = torch.linspace(self.T - 1, 0, n_steps).long().to(ctx.device)
        for i, t in enumerate(ts):
            t_b = torch.full((B,), int(t.item()), device=ctx.device)
            eps_hat = self.eps_pred(A, t_b, h)
            a_bar_t = self.alpha_bar[t]
            a_t = self.alphas[t]
            beta_t = self.betas[t]
            mean = (1 / a_t.sqrt()) * (A - (beta_t / (1 - a_bar_t).sqrt()) * eps_hat)
            if i < n_steps - 1:
                z = torch.randn_like(A)
                A = mean + beta_t.sqrt() * z
            else:
                A = mean
        return A


def train_hard(model, cfg: HardConfig, name: str):
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    for step in range(cfg.train_steps):
        emb = torch.randint(0, cfg.n_emb, (cfg.batch_size,))
        ctx = F.one_hot(emb, num_classes=cfg.n_emb).float().to(cfg.device)
        A_star, mask = make_chunk_targets(emb, cfg.horizon)
        A_star, mask = A_star.to(cfg.device), mask.to(cfg.device)
        loss = model.loss(ctx, A_star, mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 300 == 0:
            print(f"  [{name}] step {step + 1:4d}  loss={loss.item():.4f}")


@torch.no_grad()
def eval_hard(model, cfg: HardConfig, n_steps: int) -> tuple[float, float]:
    """Return (mean MSE on valid dims, walltime ms per call)."""
    n_eval = 96
    model.eval()
    emb = torch.arange(cfg.n_emb).repeat_interleave(n_eval // cfg.n_emb)
    ctx = F.one_hot(emb, num_classes=cfg.n_emb).float().to(cfg.device)
    A_star, mask = make_chunk_targets(emb, cfg.horizon)
    A_star, mask = A_star.to(cfg.device), mask.to(cfg.device)
    t0 = time.perf_counter()
    pred = model.sample(ctx, n_steps=n_steps)
    dt = (time.perf_counter() - t0) * 1000
    sq = (pred - A_star) ** 2
    mse = ((mask * sq).sum() / mask.sum().clamp_min(1.0)).item()
    return mse, dt


def run_hard() -> None:
    print("=" * 60)
    print("HARD: mini pi0 -- FM vs DDPM with multi-embodiment masking")
    print("=" * 60)
    cfg = HardConfig()
    torch.manual_seed(2)

    fm = FMPolicyMasked(cfg).to(cfg.device)
    print("\n[Train] FM head (pi0-style)")
    train_hard(fm, cfg, "FM")

    dm = DDPMPolicyMasked(cfg).to(cfg.device)
    print("\n[Train] DDPM head (Diffusion-Policy-style)")
    train_hard(dm, cfg, "DDPM")

    print("\nInference quality vs step count (masked MSE on valid dims)")
    print(f"{'method':<10}{'steps':>8}{'MSE':>14}{'wall_ms':>14}")
    print("-" * 46)
    fm_results = []
    for n in [2, 5, 10, 20, 50]:
        mse, dt = eval_hard(fm, cfg, n)
        fm_results.append((n, mse, dt))
        print(f"{'FM':<10}{n:>8}{mse:>14.5f}{dt:>14.2f}")
    dm_results = []
    for n in [10, 20, 50, 100]:
        mse, dt = eval_hard(dm, cfg, n)
        dm_results.append((n, mse, dt))
        print(f"{'DDPM':<10}{n:>8}{mse:>14.5f}{dt:>14.2f}")

    # ------- Auto commentary -------
    fm_5 = next(m for n, m, _ in fm_results if n == 5)
    dm_50 = next(m for n, m, _ in dm_results if n == 50)
    fm_5_wall = next(d for n, _, d in fm_results if n == 5)
    dm_50_wall = next(d for n, _, d in dm_results if n == 50)

    print("\nCommentary:")
    print(f"  FM @ 5 steps  -> MSE={fm_5:.4f}  walltime={fm_5_wall:.1f} ms")
    print(f"  DDPM @ 50 stp -> MSE={dm_50:.4f}  walltime={dm_50_wall:.1f} ms")
    if fm_5_wall > 0:
        ratio = dm_50_wall / fm_5_wall
        print(f"  walltime ratio DDPM(50) / FM(5) = {ratio:.1f}x")
    print("  Interpretation: even on a toy 7-d action space, FM hits comparable")
    print("  MSE in 5 Euler steps while DDPM needs 50 ancestral steps. On real")
    print("  pi0-scale (300M expert + 50-frame chunks @ 50 Hz) this is the")
    print("  difference between deployable and not. DDPM remains preferable")
    print("  when the target distribution is highly multimodal AND the action")
    print("  space is large -- but for closed-loop control, FM is the new default.\n")


# ============================================================================
# Entry point
# ============================================================================
def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("easy", "all"):
        run_easy()
    if which in ("medium", "all"):
        run_medium()
    if which in ("hard", "all"):
        run_hard()
    print("Done.")
    print("Reference: REFERENCES.md #14, Black et al., pi0, arxiv:2410.24164.")


if __name__ == "__main__":
    main()
