"""
J26 - Training loop Diffusion Policy (self-contained, < 2 min CPU).

Ce script est volontairement AUTONOME. Il ne dependant PAS des fichiers de
J24 (dataset PushT) ni de J25 (architecture ResNet18+UNet1D), parce que ce
module-jour est concu pour etre lu/joue independamment.

Pour adapter au capstone reel :
  - Remplace `make_toy_dataset()` par ton DataLoader LeRobotDataset PushT (J24).
  - Remplace `TinyDiffusionPolicy` par l'architecture ResNet18 + UNet1D
    construite a J25.
  - Le training loop lui-meme (loss MSE eps, AdamW, EMA, cosine schedule,
    gradient clip, checkpoint) reste IDENTIQUE et c'est le point pedagogique.

Stack : torch + numpy + matplotlib uniquement.
Reference : Chi et al., Diffusion Policy (RSS 2023, REFERENCES.md #19).
"""

import math
import os
from copy import deepcopy
from pathlib import Path

import matplotlib

# Backend non-interactif : on sauvegarde un PNG, on n'affiche pas de fenetre.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# 1. Dataset jouet : sequences d'actions 2D conditionnees par etat 2D
# ---------------------------------------------------------------------------
# On simule un "PushT simplifie" : etat = position courante (x, y), action =
# une sequence de 8 increments (dx, dy) pour rejoindre une cible. Cela suffit
# a illustrer la training loop ; la nature multimodale serait obtenue avec un
# dataset plus riche, mais pour un script CPU rapide on reste minimaliste.


class ToyTrajectoryDataset(Dataset):
    def __init__(self, n_samples: int = 4096, horizon: int = 8, seed: int = 0):
        rng = np.random.default_rng(seed)
        # Etat initial uniforme dans [-1, 1]^2.
        self.states = rng.uniform(-1.0, 1.0, size=(n_samples, 2)).astype(np.float32)
        # Cible aleatoire dans [-1, 1]^2.
        targets = rng.uniform(-1.0, 1.0, size=(n_samples, 2)).astype(np.float32)
        # Sequence d'actions = chemin lineaire bruite vers la cible, decoupe en
        # `horizon` increments. Resultat shape (N, horizon, 2).
        deltas = (targets - self.states) / horizon  # (N, 2)
        deltas = deltas[:, None, :].repeat(horizon, axis=1)  # (N, horizon, 2)
        # Ajout d'un peu de bruit pour rendre la regression non triviale.
        deltas += rng.normal(0.0, 0.02, size=deltas.shape).astype(np.float32)
        self.actions = deltas

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.states[idx]),  # cond, shape (2,)
            torch.from_numpy(self.actions[idx]),  # x_0, shape (horizon, 2)
        )


# ---------------------------------------------------------------------------
# 2. DDPM scheduler minimal (Ho 2020, equations 4 et 14)
# ---------------------------------------------------------------------------


class DDPMScheduler:
    """Forward noising q(x_t | x_0) avec un beta schedule lineaire.

    On expose les buffers necessaires au training :
      - alpha_bar_t : produit cumulatif des (1 - beta) jusqu'au pas t.
      - sqrt_alpha_bar et sqrt_one_minus_alpha_bar pour la reparametrisation.
    """

    def __init__(self, num_steps: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.num_steps = num_steps
        betas = torch.linspace(beta_start, beta_end, num_steps)
        alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(alphas, dim=0)  # shape (T,)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x_0 shape : (B, horizon, action_dim) ; t shape : (B,)
        # On "broadcast" sqrt_alpha_bar[t] sur les dimensions trailing.
        sqrt_ab = self.sqrt_alpha_bar.to(x_0.device)[t][:, None, None]
        sqrt_omab = self.sqrt_one_minus_alpha_bar.to(x_0.device)[t][:, None, None]
        return sqrt_ab * x_0 + sqrt_omab * noise


# ---------------------------------------------------------------------------
# 3. Mini-modele Diffusion Policy
# ---------------------------------------------------------------------------
# Architecture jouet : MLP qui prend en entree (x_t aplati, embedding sinusoidal
# de t, cond) et predit eps de meme shape que x_0. Cela suffit pour illustrer
# le training loop. Dans le vrai capstone, c'est un UNet1D ou Transformer
# conditionne par un encoder visuel ResNet18.


def sinusoidal_embedding(t: torch.Tensor, dim: int = 64) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    angles = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class TinyDiffusionPolicy(nn.Module):
    def __init__(self, action_dim: int = 2, horizon: int = 8, cond_dim: int = 2, hidden: int = 128):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        flat_dim = horizon * action_dim
        self.net = nn.Sequential(
            nn.Linear(flat_dim + 64 + cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, flat_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x_t shape (B, horizon, action_dim) -> flatten
        B = x_t.shape[0]
        x_flat = x_t.reshape(B, -1)
        t_emb = sinusoidal_embedding(t, dim=64)
        h = torch.cat([x_flat, t_emb, cond], dim=-1)
        out = self.net(h)
        return out.reshape(B, self.horizon, self.action_dim)


# ---------------------------------------------------------------------------
# 4. EMA (Exponential Moving Average) on weights
# ---------------------------------------------------------------------------


class EMA:
    """EMA simple sur l'ensemble des parametres.

    decay typique = 0.9999. Pendant un warmup on n'applique pas l'update
    (les poids initiaux random pollueraient la moyenne).
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, warmup: int = 100):
        self.decay = decay
        self.warmup = warmup
        # On stocke une copie en fp32 detachee.
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module, step: int) -> None:
        if step < self.warmup:
            # On synchronise simplement pendant le warmup.
            for k, v in model.state_dict().items():
                self.shadow[k].copy_(v.detach())
            return
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow)


# ---------------------------------------------------------------------------
# 5. Cosine LR schedule with linear warmup
# ---------------------------------------------------------------------------


def lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# 6. Training loop
# ---------------------------------------------------------------------------


def train(
    out_dir: Path,
    total_steps: int = 1500,
    batch_size: int = 64,
    lr_max: float = 1e-3,
    warmup_steps: int = 100,
    horizon: int = 8,
    num_diffusion_steps: int = 100,
    seed: int = 0,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset + loader.
    dataset = ToyTrajectoryDataset(n_samples=4096, horizon=horizon, seed=seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Modele + EMA + scheduler de bruit.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyDiffusionPolicy(action_dim=2, horizon=horizon, cond_dim=2).to(device)
    ema = EMA(model, decay=0.999, warmup=50)
    noise_sched = DDPMScheduler(num_steps=num_diffusion_steps)

    # Optimizer AdamW + LR schedule.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr_max,
        weight_decay=1e-6,
        betas=(0.95, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: lr_lambda(s, warmup_steps, total_steps),
    )

    # Boucle principale.
    losses = []
    step = 0
    data_iter = iter(loader)
    while step < total_steps:
        try:
            cond, x_0 = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            cond, x_0 = next(data_iter)
        cond = cond.to(device)
        x_0 = x_0.to(device)
        B = x_0.shape[0]

        # 1. Tirer un t aleatoire dans [0, T-1] pour chaque sample du batch.
        t = torch.randint(0, num_diffusion_steps, (B,), device=device)
        # 2. Tirer eps ~ N(0, I) de meme shape que x_0.
        eps = torch.randn_like(x_0)
        # 3. Forward noising : x_t = sqrt(alpha_bar) x_0 + sqrt(1-alpha_bar) eps.
        x_t = noise_sched.q_sample(x_0, t, eps)
        # 4. Le modele predit eps_hat.
        eps_hat = model(x_t, t, cond)
        # 5. Loss = MSE entre bruit predit et bruit vrai. C'est LE coeur DDPM.
        loss = F.mse_loss(eps_hat, eps)

        # Backward + step.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Gradient clipping : indispensable sur diffusion (gradient peut spiker).
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        ema.update(model, step)

        losses.append(loss.item())

        if step % 100 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"step {step:5d}  loss {loss.item():.4f}  lr {current_lr:.2e}")
        step += 1

    # Sauvegarde checkpoint final (modele live + EMA).
    ckpt_path = out_dir / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.shadow,
            "step": step,
            "loss_history": losses,
        },
        ckpt_path,
    )
    print(f"checkpoint sauvegarde dans {ckpt_path}")

    # Plot de la loss (courbe de training).
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, alpha=0.5, label="loss step")
    # Moving average pour lisser.
    if len(losses) > 50:
        kernel = np.ones(50) / 50
        smooth = np.convolve(losses, kernel, mode="valid")
        ax.plot(np.arange(len(smooth)) + 25, smooth, color="red", label="MA(50)")
    ax.set_xlabel("step")
    ax.set_ylabel("MSE(eps_hat, eps)")
    ax.set_title("Diffusion Policy training - toy dataset")
    ax.legend()
    fig.tight_layout()
    plot_path = out_dir / "loss_curve.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"courbe de loss sauvegardee dans {plot_path}")

    return losses, ckpt_path


# ---------------------------------------------------------------------------
# 7. Entree script
# ---------------------------------------------------------------------------


def main():
    out_dir = Path(os.environ.get("J26_OUT_DIR", "out_j26"))
    losses, ckpt = train(out_dir=out_dir, total_steps=1500, batch_size=64)
    print(f"Final loss (mean of last 100 steps): {np.mean(losses[-100:]):.4f}")
    print(f"Checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
