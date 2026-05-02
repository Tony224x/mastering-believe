"""J25 - Capstone Architecture: Diffusion Policy implementation (no training).

Builds the four blocks of Diffusion Policy as described in `01-theory/25-capstone-architecture.md`:

  1. VisionEncoder        - ResNet18 (torchvision) with a CNN fallback if torchvision is missing.
  2. StateEncoder         - 2-layer MLP turning (T_obs, state_dim) into a latent.
  3. ConditionalUNet1D    - 1D UNet with FiLM conditioning, predicts the noise eps_theta.
  4. DDPMScheduler        - cosine beta schedule + q_sample (forward) and p_sample (reverse).

Then a `DiffusionPolicy` glues everything together. We run a dummy forward pass with a
tiny batch to verify shapes (no real training here - that lands in J26).

Sources:
- REFERENCES.md #19 (Diffusion Policy, Chi et al. RSS 2023, real-stanford/diffusion_policy)
- REFERENCES.md #23 (MIT 6.S184, Holderrieth & Erives 2025) for DDPM fundamentals.

Run: `python 25-capstone-architecture.py`
Compile-check: `python -m py_compile 25-capstone-architecture.py`
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================================
# Hyperparameters (mirrors `diffusion_policy/config/policy/diffusion_policy_cnn.yaml`)
# =====================================================================================

@dataclass
class DPConfig:
    # Task shapes (PushT defaults).
    action_dim: int = 2          # (x, y) of TCP target
    state_dim: int = 2           # current TCP / object pose dims (kept tiny for the demo)
    image_size: int = 96         # H == W
    image_channels: int = 3

    # Horizons.
    obs_horizon: int = 2         # T_obs
    action_horizon: int = 16     # T_act (chunk we predict)
    action_exec_horizon: int = 8 # T_a (we execute before replanning - documentation only)

    # Diffusion.
    num_diffusion_steps: int = 100
    beta_schedule: str = "cosine"  # "cosine" or "linear"

    # Encoders / UNet.
    vision_feat_dim: int = 512   # ResNet18 fc-removed output
    state_feat_dim: int = 64
    cond_dim: int = 256
    unet_down_channels: Tuple[int, ...] = (256, 512, 1024)
    unet_kernel_size: int = 5
    n_groups: int = 8


# =====================================================================================
# 1. Vision encoder
# =====================================================================================

def _try_torchvision_resnet18(out_dim: int) -> Optional[nn.Module]:
    """Return a torchvision ResNet18 with the fc head removed, or None if torchvision missing."""
    try:
        from torchvision.models import resnet18  # noqa: WPS433 (local import is intentional)
    except Exception:  # pragma: no cover - exercised when torchvision is absent
        return None

    backbone = resnet18(weights=None)  # weights=None: don't fetch ImageNet on CI / fresh env
    # Replace fc with an identity then a projection to out_dim. ResNet18 fc input = 512.
    in_features = backbone.fc.in_features
    backbone.fc = nn.Identity()  # type: ignore[assignment]
    proj = nn.Linear(in_features, out_dim) if in_features != out_dim else nn.Identity()
    return nn.Sequential(backbone, proj)


class _SimpleCNN(nn.Module):
    """Fallback CNN with the same contract: (B, C, H, W) -> (B, out_dim)."""

    def __init__(self, in_channels: int, out_dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),  # /2
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),           # /4
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),          # /8
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),         # /16
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # global pool
            nn.Flatten(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class VisionEncoder(nn.Module):
    """Image -> vision feature vector. Tries ResNet18; falls back to a small CNN."""

    def __init__(self, cfg: DPConfig) -> None:
        super().__init__()
        backbone = _try_torchvision_resnet18(out_dim=cfg.vision_feat_dim)
        if backbone is None:
            backbone = _SimpleCNN(in_channels=cfg.image_channels, out_dim=cfg.vision_feat_dim)
        self.backbone = backbone
        self.out_dim = cfg.vision_feat_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, T_obs, C, H, W) -> (B, T_obs * vision_feat_dim)."""
        b, t, c, h, w = images.shape
        flat = images.reshape(b * t, c, h, w)
        feat = self.backbone(flat)  # (B*T, vision_feat_dim)
        feat = feat.reshape(b, t * self.out_dim)
        return feat


# =====================================================================================
# 2. State encoder
# =====================================================================================

class StateEncoder(nn.Module):
    """(B, T_obs, state_dim) -> (B, T_obs * state_feat_dim)."""

    def __init__(self, cfg: DPConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.mlp = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.state_feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.state_feat_dim, cfg.state_feat_dim),
        )
        self.out_dim = cfg.state_feat_dim

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        b, t, d = state.shape
        flat = state.reshape(b * t, d)
        feat = self.mlp(flat).reshape(b, t * self.out_dim)
        return feat


# =====================================================================================
# 3. ConditionalUNet1D (FiLM-conditioned)
# =====================================================================================

class SinusoidalPosEmb(nn.Module):
    """Standard DDPM sinusoidal embedding for diffusion timesteps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        # log-spaced frequencies (Vaswani 2017 / Ho 2020).
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=device) / max(half - 1, 1))
        emb = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class Conv1DBlock(nn.Module):
    """Conv1D + GroupNorm + Mish."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, n_groups: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_ch),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """Two Conv1D blocks with FiLM conditioning between them, plus a residual skip.

    Mirrors `ConditionalResidualBlock1D` in
    `diffusion_policy/model/diffusion/conditional_unet1d.py`.
    """

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, kernel_size: int, n_groups: int) -> None:
        super().__init__()
        self.block1 = Conv1DBlock(in_ch, out_ch, kernel_size, n_groups)
        self.block2 = Conv1DBlock(out_ch, out_ch, kernel_size, n_groups)
        # FiLM: from cond -> 2*out_ch (gamma, beta).
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_ch * 2),
        )
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.block1(x)
        film = self.cond_encoder(cond)              # (B, 2*out_ch)
        gamma, beta = film.chunk(2, dim=-1)         # each (B, out_ch)
        out = gamma.unsqueeze(-1) * out + beta.unsqueeze(-1)
        out = self.block2(out)
        return out + self.residual(x)


class Downsample1D(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConditionalUNet1D(nn.Module):
    """1D UNet that predicts noise eps_theta(a_t, t, cond).

    Input  : actions_t  (B, T_act, action_dim)
    Output : eps_pred   (B, T_act, action_dim)
    """

    def __init__(self, cfg: DPConfig, global_cond_dim: int) -> None:
        super().__init__()
        self.cfg = cfg

        # Time embedding -> mixed with global cond.
        time_dim = cfg.cond_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )
        # Project (image+state) global cond down to cond_dim, then merge with time.
        self.cond_proj = nn.Linear(global_cond_dim, cfg.cond_dim)
        cond_dim_film = cfg.cond_dim  # the cond dim that FiLM blocks consume

        # Channel ladder: action_dim -> down_channels.
        ch_in = cfg.action_dim
        ch_list = [ch_in, *cfg.unet_down_channels]
        kernel = cfg.unet_kernel_size
        n_groups = cfg.n_groups

        # Down path.
        self.down_blocks = nn.ModuleList()
        for i in range(len(ch_list) - 1):
            ch_a, ch_b = ch_list[i], ch_list[i + 1]
            is_last = (i == len(ch_list) - 2)
            self.down_blocks.append(nn.ModuleList([
                ConditionalResidualBlock1D(ch_a, ch_b, cond_dim_film, kernel, n_groups),
                ConditionalResidualBlock1D(ch_b, ch_b, cond_dim_film, kernel, n_groups),
                Downsample1D(ch_b) if not is_last else nn.Identity(),
            ]))

        mid_ch = ch_list[-1]
        self.mid_block1 = ConditionalResidualBlock1D(mid_ch, mid_ch, cond_dim_film, kernel, n_groups)
        self.mid_block2 = ConditionalResidualBlock1D(mid_ch, mid_ch, cond_dim_film, kernel, n_groups)

        # Up path (mirrors down).
        self.up_blocks = nn.ModuleList()
        ch_list_up = list(reversed(ch_list))  # e.g. [1024, 512, 256, action_dim]
        for i in range(len(ch_list_up) - 1):
            ch_a, ch_b = ch_list_up[i], ch_list_up[i + 1]
            is_last = (i == len(ch_list_up) - 2)
            # After concat with skip, in-channels double.
            self.up_blocks.append(nn.ModuleList([
                ConditionalResidualBlock1D(ch_a * 2, ch_a, cond_dim_film, kernel, n_groups),
                ConditionalResidualBlock1D(ch_a, ch_b, cond_dim_film, kernel, n_groups),
                Upsample1D(ch_b) if not is_last else nn.Identity(),
            ]))

        # Final conv to action_dim.
        self.final_conv = nn.Conv1d(cfg.action_dim, cfg.action_dim, 1)

    def forward(
        self,
        action: torch.Tensor,        # (B, T_act, A)
        timestep: torch.Tensor,      # (B,) int64
        global_cond: torch.Tensor,   # (B, global_cond_dim)
    ) -> torch.Tensor:
        # (B, T_act, A) -> (B, A, T_act) for Conv1D over the time axis.
        x = action.transpose(1, 2)

        t_emb = self.time_mlp(timestep)             # (B, cond_dim)
        c_emb = self.cond_proj(global_cond)         # (B, cond_dim)
        cond = t_emb + c_emb                        # mix time + global cond

        skips = []
        for res1, res2, down in self.down_blocks:
            x = res1(x, cond)
            x = res2(x, cond)
            skips.append(x)
            x = down(x)

        x = self.mid_block1(x, cond)
        x = self.mid_block2(x, cond)

        for res1, res2, up in self.up_blocks:
            skip = skips.pop()
            # Spatial sizes might mismatch by 1 due to odd action_horizon; pad/crop if needed.
            if x.shape[-1] != skip.shape[-1]:
                diff = skip.shape[-1] - x.shape[-1]
                if diff > 0:
                    x = F.pad(x, (0, diff))
                else:
                    x = x[..., :skip.shape[-1]]
            x = torch.cat([x, skip], dim=1)
            x = res1(x, cond)
            x = res2(x, cond)
            x = up(x)

        x = self.final_conv(x)
        return x.transpose(1, 2)  # back to (B, T_act, A)


# =====================================================================================
# 4. DDPM scheduler (cosine beta schedule)
# =====================================================================================

def cosine_beta_schedule(num_steps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule from Nichol & Dhariwal 2021 (Improved DDPM)."""
    steps = num_steps + 1
    t = torch.linspace(0, num_steps, steps, dtype=torch.float64) / num_steps
    alpha_bar = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clip(betas, 0.0, 0.999).float()


def linear_beta_schedule(num_steps: int) -> torch.Tensor:
    return torch.linspace(1e-4, 0.02, num_steps)


class DDPMScheduler:
    """Minimal DDPM scheduler. Holds beta/alpha tables and exposes q_sample / step."""

    def __init__(self, cfg: DPConfig, device: torch.device | str = "cpu") -> None:
        self.cfg = cfg
        self.num_steps = cfg.num_diffusion_steps
        if cfg.beta_schedule == "cosine":
            betas = cosine_beta_schedule(self.num_steps)
        elif cfg.beta_schedule == "linear":
            betas = linear_beta_schedule(self.num_steps)
        else:
            raise ValueError(f"Unknown beta_schedule: {cfg.beta_schedule}")

        self.betas = betas.to(device)
        self.alphas = (1.0 - self.betas)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # Pre-compute helpers used in q_sample / p_sample.
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward process: a_t = sqrt(alpha_bar_t)*a_0 + sqrt(1-alpha_bar_t)*eps."""
        sqrt_ab = self.sqrt_alphas_cumprod[t].view(-1, *([1] * (x0.dim() - 1)))
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].view(-1, *([1] * (x0.dim() - 1)))
        return sqrt_ab * x0 + sqrt_om * noise

    @torch.no_grad()
    def step(self, eps_pred: torch.Tensor, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """One reverse step a_t -> a_{t-1} (DDPM sampling)."""
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alphas_cumprod[t]
        # mu_theta = (1/sqrt(alpha_t)) * (a_t - (beta_t / sqrt(1-alpha_bar_t)) * eps_pred)
        coef = beta_t / torch.sqrt(1.0 - alpha_bar_t)
        mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - coef * eps_pred)
        if t > 0:
            noise = torch.randn_like(x_t)
            sigma_t = torch.sqrt(beta_t)
            return mean + sigma_t * noise
        return mean


# =====================================================================================
# 5. DiffusionPolicy: glue it all together
# =====================================================================================

class DiffusionPolicy(nn.Module):
    """End-to-end Diffusion Policy: encoders + UNet 1D + DDPM scheduler.

    Two key entry points:
      - `compute_loss(batch)`        -> J26 will train against this.
      - `predict_action(batch)`      -> J27 will eval rollouts via this.
    """

    def __init__(self, cfg: DPConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vision_encoder = VisionEncoder(cfg)
        self.state_encoder = StateEncoder(cfg)
        # Global cond dim = T_obs * (vision_feat_dim + state_feat_dim).
        self.global_cond_dim = cfg.obs_horizon * (cfg.vision_feat_dim + cfg.state_feat_dim)
        self.denoiser = ConditionalUNet1D(cfg, global_cond_dim=self.global_cond_dim)
        self.scheduler = DDPMScheduler(cfg)

    # ---- conditioning ---------------------------------------------------------------

    def encode_obs(self, obs_image: torch.Tensor, obs_state: torch.Tensor) -> torch.Tensor:
        """obs_image: (B, T_obs, C, H, W); obs_state: (B, T_obs, state_dim) -> (B, cond_global)."""
        f_img = self.vision_encoder(obs_image)
        f_st = self.state_encoder(obs_state)
        return torch.cat([f_img, f_st], dim=-1)

    # ---- training loss --------------------------------------------------------------

    def compute_loss(self, batch: dict) -> torch.Tensor:
        """`batch` keys: `obs_image` (B,T_obs,C,H,W), `obs_state` (B,T_obs,state_dim),
        `actions` (B, T_act, action_dim).
        """
        actions = batch["actions"]
        b = actions.shape[0]
        cond = self.encode_obs(batch["obs_image"], batch["obs_state"])
        t = torch.randint(0, self.scheduler.num_steps, (b,), device=actions.device)
        noise = torch.randn_like(actions)
        a_t = self.scheduler.q_sample(actions, t, noise)
        eps_pred = self.denoiser(a_t, t, cond)
        return F.mse_loss(eps_pred, noise)

    # ---- inference ------------------------------------------------------------------

    @torch.no_grad()
    def predict_action(self, batch: dict) -> torch.Tensor:
        """Reverse diffusion, returns (B, T_act, action_dim)."""
        cond = self.encode_obs(batch["obs_image"], batch["obs_state"])
        b = cond.shape[0]
        device = cond.device
        a = torch.randn(b, self.cfg.action_horizon, self.cfg.action_dim, device=device)
        for t in reversed(range(self.scheduler.num_steps)):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            eps_pred = self.denoiser(a, t_batch, cond)
            a = self.scheduler.step(eps_pred, a, t)
        return a


# =====================================================================================
# 6. Sanity check: dummy forward pass (no training!)
# =====================================================================================

def make_fake_batch(cfg: DPConfig, batch_size: int = 4) -> dict:
    """Tiny synthetic batch matching the dataset contract from J24."""
    rng = np.random.default_rng(0)
    obs_image = torch.from_numpy(
        rng.standard_normal((batch_size, cfg.obs_horizon, cfg.image_channels, cfg.image_size, cfg.image_size)).astype(np.float32)
    )
    obs_state = torch.from_numpy(
        rng.standard_normal((batch_size, cfg.obs_horizon, cfg.state_dim)).astype(np.float32)
    )
    actions = torch.from_numpy(
        rng.standard_normal((batch_size, cfg.action_horizon, cfg.action_dim)).astype(np.float32)
    )
    return {"obs_image": obs_image, "obs_state": obs_state, "actions": actions}


def main() -> None:
    cfg = DPConfig()
    print("== DiffusionPolicy capstone (J25) ==")
    print(f"action_dim={cfg.action_dim} action_horizon={cfg.action_horizon} "
          f"obs_horizon={cfg.obs_horizon} num_diffusion_steps={cfg.num_diffusion_steps} "
          f"beta_schedule={cfg.beta_schedule}")

    policy = DiffusionPolicy(cfg)
    n_params = sum(p.numel() for p in policy.parameters())
    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"params: total={n_params:,} trainable={n_trainable:,}")

    batch = make_fake_batch(cfg, batch_size=2)
    print({k: tuple(v.shape) for k, v in batch.items()})

    # Training-style forward (computes loss, does NOT optimize).
    policy.train()
    loss = policy.compute_loss(batch)
    print(f"compute_loss -> {loss.item():.4f} (should be ~1.0 since model is untrained)")

    # Inference-style forward: full reverse diffusion. We use a tiny step count
    # for the smoke test, otherwise CPU time blows up.
    cfg_small = DPConfig(num_diffusion_steps=4)
    policy_small = DiffusionPolicy(cfg_small)
    policy_small.eval()
    batch_small = make_fake_batch(cfg_small, batch_size=2)
    actions = policy_small.predict_action(batch_small)
    print(f"predict_action shape -> {tuple(actions.shape)}  "
          f"(expected (2, {cfg_small.action_horizon}, {cfg_small.action_dim}))")
    assert actions.shape == (2, cfg_small.action_horizon, cfg_small.action_dim)

    print("OK - architecture wired, shapes consistent, ready for J26 training loop.")


if __name__ == "__main__":
    main()
