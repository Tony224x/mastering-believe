"""Solutions for J25 exercises (easy/medium/hard).

Run any block independently:
    python 25-capstone-architecture.py easy
    python 25-capstone-architecture.py medium
    python 25-capstone-architecture.py hard-transformer
    python 25-capstone-architecture.py hard-cfg

Imports the J25 architecture from `domains/robotics-ai/02-code/25-capstone-architecture.py`.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------------
# Robust import of the J25 module by file path (no package needed).
# ---------------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
DOMAIN_ROOT = THIS_FILE.parents[2]            # .../domains/robotics-ai
J25_PATH = DOMAIN_ROOT / "02-code" / "25-capstone-architecture.py"


def _load_j25():
    spec = importlib.util.spec_from_file_location("j25_arch", J25_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load J25 module from {J25_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# =================================================================================
# EASY: count parameters + verify forward shapes.
# =================================================================================

def solve_easy() -> None:
    j25 = _load_j25()
    cfg = j25.DPConfig()
    policy = j25.DiffusionPolicy(cfg)

    def count(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters())

    n_vision = count(policy.vision_encoder)
    n_state = count(policy.state_encoder)
    n_denoiser = count(policy.denoiser)
    n_total = count(policy)
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)

    print("== EASY ==")
    print(f"vision_encoder : {n_vision:>12,}")
    print(f"state_encoder  : {n_state:>12,}")
    print(f"denoiser (UNet): {n_denoiser:>12,}")
    print(f"total          : {n_total:>12,}  trainable: {n_train:,}")

    batch = j25.make_fake_batch(cfg, batch_size=3)
    loss = policy.compute_loss(batch)
    print(f"compute_loss(B=3) -> {loss.item():.4f}  (>=0: {loss.item() >= 0})")

    # Use a tiny step count so CPU sampling is fast in the smoke test.
    cfg_small = j25.DPConfig(num_diffusion_steps=4)
    policy_small = j25.DiffusionPolicy(cfg_small)
    policy_small.eval()
    batch_small = j25.make_fake_batch(cfg_small, batch_size=3)
    actions = policy_small.predict_action(batch_small)
    expected = (3, cfg_small.action_horizon, cfg_small.action_dim)
    print(f"predict_action shape -> {tuple(actions.shape)} (expected {expected})")
    assert tuple(actions.shape) == expected

    print(
        "Reponse (typique) : le denoiser est de loin le plus lourd (>90% des params), "
        "principalement car ses canaux montent jusqu'a 1024 dans le bottleneck du UNet 1D."
    )


# =================================================================================
# MEDIUM: DDIM scheduler + comparison with DDPM.
# =================================================================================

class DDIMScheduler:
    """DDIM (Song 2020), eta=0 deterministic. Reuses the same betas as DDPMScheduler.

    `step_ddim` follows:
        x0_hat        = (x_t - sqrt(1-alpha_bar_t) * eps) / sqrt(alpha_bar_t)
        x_{t_prev}    = sqrt(alpha_bar_{t_prev}) * x0_hat
                        + sqrt(1 - alpha_bar_{t_prev}) * eps   (eta=0 case)
    """

    def __init__(self, ddpm) -> None:
        self.alphas_cumprod = ddpm.alphas_cumprod
        self.num_steps = ddpm.num_steps

    def make_timesteps(self, n_steps: int) -> list[int]:
        # Evenly spaced timesteps from T-1 down to 0.
        ts = torch.linspace(self.num_steps - 1, 0, n_steps).round().long().tolist()
        return ts

    @torch.no_grad()
    def step_ddim(self, eps: torch.Tensor, x_t: torch.Tensor, t: int, t_prev: int) -> torch.Tensor:
        ab_t = self.alphas_cumprod[t]
        ab_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=eps.device)
        x0 = (x_t - torch.sqrt(1.0 - ab_t) * eps) / torch.sqrt(ab_t)
        return torch.sqrt(ab_prev) * x0 + torch.sqrt(1.0 - ab_prev) * eps


def _identity_denoise(eps: torch.Tensor) -> torch.Tensor:
    """Stand-in for a perfectly trained denoiser: returns the noise it was given.

    Since we don't actually train here, we use the *true* noise as `eps_pred`.
    With a perfect predictor, both schedulers must reconstruct `a_0` near-perfectly
    when run through the full schedule, so any residual MSE is purely a function
    of the scheduler discretization.
    """
    return eps


def solve_medium() -> None:
    j25 = _load_j25()
    cfg = j25.DPConfig(num_diffusion_steps=100)
    ddpm = j25.DDPMScheduler(cfg)
    ddim = DDIMScheduler(ddpm)

    torch.manual_seed(0)
    a0 = torch.randn(1, cfg.action_horizon, cfg.action_dim)
    eps_true = torch.randn_like(a0)
    t_T = cfg.num_diffusion_steps - 1
    a_T = ddpm.q_sample(a0, torch.tensor([t_T]), eps_true)

    rows = []

    # DDPM full reverse (we know eps_true so use it as eps_pred).
    t0 = time.perf_counter()
    a = a_T.clone()
    for t in reversed(range(cfg.num_diffusion_steps)):
        a = ddpm.step(_identity_denoise(eps_true), a, t)
    dt = time.perf_counter() - t0
    rows.append(("DDPM", cfg.num_diffusion_steps, F.mse_loss(a, a0).item(), dt))

    for n in (50, 25, 10):
        ts = ddim.make_timesteps(n)
        t0 = time.perf_counter()
        a = a_T.clone()
        for i, t in enumerate(ts):
            t_prev = ts[i + 1] if i + 1 < len(ts) else -1
            a = ddim.step_ddim(_identity_denoise(eps_true), a, t, t_prev)
        dt = time.perf_counter() - t0
        rows.append(("DDIM", n, F.mse_loss(a, a0).item(), dt))

    print("== MEDIUM ==")
    print(f"{'Scheduler':<10}{'Steps':>8}{'MSE vs a0':>14}{'Time (s)':>12}")
    for name, n, mse, t in rows:
        print(f"{name:<10}{n:>8}{mse:>14.6f}{t:>12.4f}")

    print(
        "\nLecture : avec un denoiser parfait, DDPM (100 steps) reconstruit ~exactement, "
        "DDIM 25 steps perd peu, DDIM 10 commence a degrader. En pratique avec un vrai modele "
        "entraine, DDIM 16 steps est l'option par defaut de Diffusion Policy."
    )


# =================================================================================
# HARD A: Transformer denoiser.
# =================================================================================

class TransformerDenoiser(nn.Module):
    """Transformer-based replacement for ConditionalUNet1D. Same I/O contract."""

    def __init__(self, cfg, global_cond_dim: int, d_model: int = 256, nhead: int = 4, n_layers: int = 4) -> None:
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.action_proj = nn.Linear(cfg.action_dim, d_model)
        self.action_unproj = nn.Linear(d_model, cfg.action_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.action_horizon, d_model))
        self.cond_proj = nn.Linear(global_cond_dim, d_model)
        # Sinusoidal time embedding -> d_model.
        self.time_proj = nn.Sequential(
            self._sinusoidal(d_model), nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    @staticmethod
    def _sinusoidal(dim: int) -> nn.Module:
        class Sin(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.d = d

            def forward(self, t):
                half = self.d // 2
                freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / max(half - 1, 1))
                emb = t[:, None].float() * freqs[None, :]
                return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        return Sin(dim)

    def forward(self, action, timestep, global_cond):
        b, t_act, _ = action.shape
        x = self.action_proj(action) + self.pos_emb[:, :t_act]
        time_emb = self.time_proj(timestep)              # (B, d_model)
        cond = self.cond_proj(global_cond) + time_emb    # (B, d_model)
        # Prepend cond as a "CLS-like" token so attention sees the conditioning.
        x = torch.cat([cond.unsqueeze(1), x], dim=1)
        x = self.encoder(x)
        x = x[:, 1:, :]                                  # drop the cond token
        return self.action_unproj(x)


def solve_hard_transformer() -> None:
    j25 = _load_j25()
    cfg = j25.DPConfig()
    policy = j25.DiffusionPolicy(cfg)

    # Swap denoiser.
    policy.denoiser = TransformerDenoiser(cfg, global_cond_dim=policy.global_cond_dim)

    n_params = sum(p.numel() for p in policy.denoiser.parameters())
    print("== HARD A (Transformer) ==")
    print(f"transformer denoiser params: {n_params:,}")

    batch = j25.make_fake_batch(cfg, batch_size=2)
    loss = policy.compute_loss(batch)
    print(f"compute_loss -> {loss.item():.4f} (must be finite)")
    assert torch.isfinite(loss)

    cfg_small = j25.DPConfig(num_diffusion_steps=4)
    policy_small = j25.DiffusionPolicy(cfg_small)
    policy_small.denoiser = TransformerDenoiser(cfg_small, global_cond_dim=policy_small.global_cond_dim)
    policy_small.eval()
    batch_small = j25.make_fake_batch(cfg_small, batch_size=2)
    actions = policy_small.predict_action(batch_small)
    print(f"predict_action shape -> {tuple(actions.shape)}")
    assert actions.shape == (2, cfg_small.action_horizon, cfg_small.action_dim)
    print(
        "Discussion : O(T_act^2) negligeable a T_act=16. UNet 1D reste preferable sur "
        "PushT car les inductive biases convolutionnels matchent le caractere local des "
        "trajectoires d'actions a cet horizon court."
    )


# =================================================================================
# HARD B: Classifier-Free Guidance.
# =================================================================================

class CFGPolicy(nn.Module):
    """Wraps a J25 DiffusionPolicy to support classifier-free guidance at inference."""

    def __init__(self, base_policy: nn.Module, p_uncond: float = 0.1) -> None:
        super().__init__()
        self.base = base_policy
        self.p_uncond = p_uncond
        # Learnable null token replacing the global_cond when dropping out.
        self.null_cond = nn.Parameter(torch.zeros(self.base.global_cond_dim))

    def _maybe_drop(self, cond: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p_uncond <= 0:
            return cond
        b = cond.shape[0]
        mask = (torch.rand(b, device=cond.device) < self.p_uncond).float().unsqueeze(-1)
        null = self.null_cond[None, :].expand_as(cond)
        return mask * null + (1 - mask) * cond

    def compute_loss(self, batch: dict) -> torch.Tensor:
        actions = batch["actions"]
        b = actions.shape[0]
        cond = self.base.encode_obs(batch["obs_image"], batch["obs_state"])
        cond = self._maybe_drop(cond)
        t = torch.randint(0, self.base.scheduler.num_steps, (b,), device=actions.device)
        noise = torch.randn_like(actions)
        a_t = self.base.scheduler.q_sample(actions, t, noise)
        eps_pred = self.base.denoiser(a_t, t, cond)
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def sample(self, batch: dict, guidance_scale: float = 1.0) -> torch.Tensor:
        cond = self.base.encode_obs(batch["obs_image"], batch["obs_state"])
        b = cond.shape[0]
        device = cond.device
        null = self.null_cond[None, :].expand_as(cond)
        a = torch.randn(b, self.base.cfg.action_horizon, self.base.cfg.action_dim, device=device)
        for t in reversed(range(self.base.scheduler.num_steps)):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            eps_uncond = self.base.denoiser(a, t_batch, null)
            eps_cond = self.base.denoiser(a, t_batch, cond)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            a = self.base.scheduler.step(eps, a, t)
        return a


def solve_hard_cfg() -> None:
    j25 = _load_j25()
    cfg = j25.DPConfig(num_diffusion_steps=4)  # tiny so the loop is fast on CPU
    base = j25.DiffusionPolicy(cfg)
    policy = CFGPolicy(base, p_uncond=0.1)

    # 1 epoch of toy training to get *some* signal in the null token.
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    batch = j25.make_fake_batch(cfg, batch_size=8)
    policy.train()
    for _ in range(20):
        opt.zero_grad()
        loss = policy.compute_loss(batch)
        loss.backward()
        opt.step()
    print("== HARD B (CFG) ==")
    print(f"final toy loss: {loss.item():.4f}")

    # Variance vs guidance_scale on N=16 samples for the same conditioning.
    policy.eval()
    fixed_batch = j25.make_fake_batch(cfg, batch_size=1)
    n_samples = 8
    rows = []
    for w in (0.0, 1.0, 2.0, 5.0):
        samples = []
        for _ in range(n_samples):
            samples.append(policy.sample(fixed_batch, guidance_scale=w))
        stack = torch.cat(samples, dim=0)  # (n_samples, T_act, A)
        var = stack.var(dim=0).mean().item()
        rows.append((w, var))
    print(f"{'w':>6}{'mean variance':>20}")
    for w, v in rows:
        print(f"{w:>6.1f}{v:>20.6f}")
    print(
        "Lecture attendue : variance qui DECROIT (ou plafonne) quand w augmente. "
        "Risque : sur-conditioning -> mode collapse (toutes les actions deviennent quasi identiques)."
    )


# =================================================================================
# Dispatcher
# =================================================================================

def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "easy"
    if cmd == "easy":
        solve_easy()
    elif cmd == "medium":
        solve_medium()
    elif cmd == "hard-transformer":
        solve_hard_transformer()
    elif cmd == "hard-cfg":
        solve_hard_cfg()
    else:
        print(f"Unknown command: {cmd}. Use easy | medium | hard-transformer | hard-cfg")
        sys.exit(2)


if __name__ == "__main__":
    main()
