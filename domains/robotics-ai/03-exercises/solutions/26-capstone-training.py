"""
J26 - Solutions des exercices easy / medium / hard.

Chaque section est independante. Toutes sont auto-suffisantes (re-imports
locaux, dataset jouet inline). Lance la fonction que tu veux verifier.

Cf. domains/robotics-ai/02-code/26-capstone-training.py pour le training loop
de reference, et REFERENCES.md #19 (Diffusion Policy, Chi 2023) pour la
config officielle.
"""

import math
import time
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ===========================================================================
# Helpers communs (reutilises par les 3 solutions)
# ===========================================================================


class ToyTrajectoryDataset(Dataset):
    def __init__(self, n_samples: int = 4096, horizon: int = 8, action_dim: int = 2, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.states = rng.uniform(-1.0, 1.0, size=(n_samples, 2)).astype(np.float32)
        targets = rng.uniform(-1.0, 1.0, size=(n_samples, action_dim)).astype(np.float32)
        # On replique l'etat sur action_dim si action_dim != 2 pour rester coherent.
        start = np.tile(self.states[:, :1], (1, action_dim))[:, :action_dim]
        deltas = (targets - start) / horizon
        deltas = deltas[:, None, :].repeat(horizon, axis=1)
        deltas += rng.normal(0.0, 0.02, size=deltas.shape).astype(np.float32)
        self.actions = deltas

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return torch.from_numpy(self.states[idx]), torch.from_numpy(self.actions[idx])


class DDPMScheduler:
    def __init__(self, num_steps=100):
        betas = torch.linspace(1e-4, 0.02, num_steps)
        alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(alphas, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
        self.num_steps = num_steps

    def q_sample(self, x_0, t, noise):
        sqrt_ab = self.sqrt_alpha_bar.to(x_0.device)[t][:, None, None]
        sqrt_omab = self.sqrt_one_minus_alpha_bar.to(x_0.device)[t][:, None, None]
        return sqrt_ab * x_0 + sqrt_omab * noise


def sinusoidal_embedding(t, dim=64):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    angles = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class TinyPolicy(nn.Module):
    def __init__(self, action_dim=2, horizon=8, cond_dim=2, hidden=128):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        flat = horizon * action_dim
        self.net = nn.Sequential(
            nn.Linear(flat + 64 + cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, flat),
        )

    def forward(self, x_t, t, cond):
        B = x_t.shape[0]
        x_flat = x_t.reshape(B, -1)
        h = torch.cat([x_flat, sinusoidal_embedding(t), cond], dim=-1)
        return self.net(h).reshape(B, self.horizon, self.action_dim)


# ===========================================================================
# Solution EASY — explorer le LR
# ===========================================================================


def solution_easy(out_dir: Path = Path("out_j26_easy"), total_steps: int = 1500):
    """Lance trois trainings avec des LR differents et trace les courbes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lrs = [1e-5, 1e-3, 1e-1]
    all_losses = {}

    for lr_max in lrs:
        torch.manual_seed(0)
        np.random.seed(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ds = ToyTrajectoryDataset()
        loader = DataLoader(ds, batch_size=64, shuffle=True, drop_last=True)
        model = TinyPolicy().to(device)
        sched_noise = DDPMScheduler()
        opt = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=1e-6, betas=(0.95, 0.999))

        warmup = 100

        def lr_fn(step, lm=lr_max):
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_fn)

        losses = []
        step = 0
        diverged = False
        data_iter = iter(loader)
        while step < total_steps and not diverged:
            try:
                cond, x_0 = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                cond, x_0 = next(data_iter)
            cond = cond.to(device)
            x_0 = x_0.to(device)
            B = x_0.shape[0]
            t = torch.randint(0, sched_noise.num_steps, (B,), device=device)
            eps = torch.randn_like(x_0)
            x_t = sched_noise.q_sample(x_0, t, eps)
            eps_hat = model(x_t, t, cond)
            loss = F.mse_loss(eps_hat, eps)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()

            l = loss.item()
            losses.append(l)
            if not math.isfinite(l):
                # Divergence reperee : on stoppe ce run pour eviter du bruit.
                diverged = True
            step += 1

        all_losses[lr_max] = losses
        np.save(out_dir / f"losses_lr_{lr_max:.0e}.npy", np.array(losses))
        print(f"LR {lr_max:.0e} : {len(losses)} steps, last loss = {losses[-1]:.4f} (diverged={diverged})")

    # Plot comparatif (axe Y log).
    fig, ax = plt.subplots(figsize=(9, 5))
    for lr_max, losses in all_losses.items():
        # Filtre les NaN avant le plot.
        arr = np.array(losses, dtype=np.float64)
        mask = np.isfinite(arr)
        ax.plot(np.where(mask)[0], arr[mask], label=f"lr={lr_max:.0e}", alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("MSE loss (log)")
    ax.set_title("Easy : impact du LR sur le training Diffusion Policy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "compare_lrs.png", dpi=120)
    plt.close(fig)
    return all_losses


# ===========================================================================
# Solution MEDIUM — EMA from scratch avec halflife
# ===========================================================================


class EMAHalflife:
    """EMA avec halflife configurable. Poids stockes en fp32.

    halflife = nombre de steps apres lequel un poids "ancien" perd 50% de
    son influence dans la moyenne. Relation : decay = 0.5 ** (1 / halflife).
    """

    def __init__(self, model: nn.Module, halflife_steps: int, warmup: int = 100):
        assert halflife_steps > 0
        # Conversion halflife -> decay.
        self.decay = 0.5 ** (1.0 / halflife_steps)
        self.warmup = warmup
        # Stockage en fp32 pour eviter la derive numerique sous fp16.
        self.shadow = {
            k: v.detach().float().clone() for k, v in model.state_dict().items()
        }

    def update(self, model: nn.Module, step: int) -> None:
        with torch.no_grad():
            if step < self.warmup:
                # Pendant le warmup on synchronise sans appliquer la formule EMA.
                for k, v in model.state_dict().items():
                    self.shadow[k].copy_(v.detach().float())
                return
            for k, v in model.state_dict().items():
                live = v.detach().float()
                self.shadow[k].mul_(self.decay).add_(live, alpha=1.0 - self.decay)

    def copy_to(self, model: nn.Module) -> None:
        # On caste vers le dtype du parametre live (utile si le modele est en fp16).
        target_state = {}
        live = model.state_dict()
        for k, v in self.shadow.items():
            target_state[k] = v.to(dtype=live[k].dtype)
        model.load_state_dict(target_state)


def _train_and_get_model(halflife_steps: int, total_steps: int = 1500):
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ToyTrajectoryDataset()
    loader = DataLoader(ds, batch_size=64, shuffle=True, drop_last=True)
    model = TinyPolicy().to(device)
    ema = EMAHalflife(model, halflife_steps=halflife_steps, warmup=50)
    sched_noise = DDPMScheduler()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6, betas=(0.95, 0.999))
    warmup = 100

    def lr_fn(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_fn)

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
        t = torch.randint(0, sched_noise.num_steps, (B,), device=device)
        eps = torch.randn_like(x_0)
        x_t = sched_noise.q_sample(x_0, t, eps)
        eps_hat = model(x_t, t, cond)
        loss = F.mse_loss(eps_hat, eps)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        ema.update(model, step)
        step += 1
    return model, ema, device


@torch.no_grad()
def _sample_from_ema(model: nn.Module, ema: EMAHalflife, cond: torch.Tensor, sched_noise: DDPMScheduler):
    """Echantillonne via DDPM reverse en utilisant les poids EMA, puis restaure les live."""
    # Sauve les poids live.
    live_state = deepcopy(model.state_dict())
    ema.copy_to(model)
    model.eval()
    device = cond.device
    B = cond.shape[0]
    horizon = model.horizon
    action_dim = model.action_dim
    # Initialisation : x_T ~ N(0, I).
    x = torch.randn(B, horizon, action_dim, device=device)
    # Reverse DDPM tres simplifie (sans variance reelle, juste illustration).
    for t_int in reversed(range(sched_noise.num_steps)):
        t = torch.full((B,), t_int, device=device, dtype=torch.long)
        eps_hat = model(x, t, cond)
        ab = sched_noise.alpha_bar.to(device)[t_int]
        ab_prev = sched_noise.alpha_bar.to(device)[t_int - 1] if t_int > 0 else torch.tensor(1.0, device=device)
        # x_0_hat = (x - sqrt(1-ab) eps_hat) / sqrt(ab)
        x_0_hat = (x - torch.sqrt(1.0 - ab) * eps_hat) / torch.sqrt(ab)
        # Reformule x vers t-1 : moyenne uniquement (sans bruit ajoute).
        x = torch.sqrt(ab_prev) * x_0_hat + torch.sqrt(1.0 - ab_prev) * eps_hat
    # Restaure les poids live.
    model.load_state_dict(live_state)
    model.train()
    return x


def solution_medium(out_dir: Path = Path("out_j26_medium"), total_steps: int = 1500):
    out_dir.mkdir(parents=True, exist_ok=True)
    halflives = [50, 500, 5000]
    sched_noise = DDPMScheduler()
    fig, axes = plt.subplots(1, len(halflives), figsize=(15, 4))
    for i, hl in enumerate(halflives):
        model, ema, device = _train_and_get_model(halflife_steps=hl, total_steps=total_steps)
        # 4 etats fixes pour echantillonner.
        cond = torch.tensor(
            [[-0.8, -0.8], [0.8, -0.8], [-0.8, 0.8], [0.8, 0.8]],
            dtype=torch.float32,
            device=device,
        )
        x_sampled = _sample_from_ema(model, ema, cond, sched_noise)  # (4, 8, 2)
        # Plot des 4 trajectoires.
        ax = axes[i]
        for j in range(4):
            traj = x_sampled[j].cpu().numpy()
            ax.plot(traj[:, 0], traj[:, 1], "-o", alpha=0.7, label=f"cond {j}")
        ax.set_title(f"halflife = {hl}")
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)
        # Sanity check : verifie que les poids EMA sont en fp32.
        sample_key = next(iter(ema.shadow))
        assert ema.shadow[sample_key].dtype == torch.float32, "EMA doit etre en fp32"
    fig.suptitle("Medium : effet du halflife EMA sur la qualite d'echantillonnage")
    fig.tight_layout()
    fig.savefig(out_dir / "ema_halflife_compare.png", dpi=120)
    plt.close(fig)
    print("Solution MEDIUM : poids EMA verifies en fp32 pour tous les halflives.")


# ===========================================================================
# Solution HARD — mixed precision + grad checkpointing + benchmark
# ===========================================================================


class MediumPolicy(nn.Module):
    def __init__(self, action_dim=4, horizon=16, hidden=1024):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        # Pseudo encoder visuel : 3 Conv2d sur (3, 64, 64).
        self.vision = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # (16, 32, 32)
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # (32, 16, 16)
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # (64, 8, 8)
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),  # (64, 1, 1)
            nn.Flatten(),  # (64,)
        )
        flat = horizon * action_dim
        in_dim = flat + 64 + 64  # x_t flat + t_emb(64) + vision(64)
        layers = []
        prev = in_dim
        for _ in range(8):
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.SiLU())
            prev = hidden
        layers.append(nn.Linear(prev, flat))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t, t, image):
        B = x_t.shape[0]
        x_flat = x_t.reshape(B, -1)
        v = self.vision(image)
        h = torch.cat([x_flat, sinusoidal_embedding(t), v], dim=-1)
        return self.net(h).reshape(B, self.horizon, self.action_dim)


class MediumPolicyCkpt(MediumPolicy):
    """Variante avec gradient checkpointing sur 2 sous-blocs du MLP."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # On split self.net en 3 chunks pour pouvoir checkpointer les 2 premiers.
        children = list(self.net.children())
        # Chunk 1 : layers 0..5 (3 Linear + 3 SiLU)
        self.chunk1 = nn.Sequential(*children[:6])
        self.chunk2 = nn.Sequential(*children[6:12])
        self.chunk3 = nn.Sequential(*children[12:])

    def forward(self, x_t, t, image):
        from torch.utils.checkpoint import checkpoint

        B = x_t.shape[0]
        x_flat = x_t.reshape(B, -1)
        v = self.vision(image)
        h = torch.cat([x_flat, sinusoidal_embedding(t), v], dim=-1)
        h = checkpoint(self.chunk1, h, use_reentrant=False)
        h = checkpoint(self.chunk2, h, use_reentrant=False)
        h = self.chunk3(h)
        return h.reshape(B, self.horizon, self.action_dim)


def _bench_one(variant: str, total_steps: int = 200, batch_size: int = 16):
    """Benchmark une variante. Retourne (throughput, peak_mb, final_loss)."""
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    if variant == "bf16+ckpt":
        model = MediumPolicyCkpt().to(device)
    else:
        model = MediumPolicy().to(device)

    sched_noise = DDPMScheduler()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6, betas=(0.95, 0.999))

    use_scaler = variant == "fp16"
    scaler = torch.amp.GradScaler(device.type) if use_scaler else None

    if variant == "fp16":
        amp_dtype = torch.float16
    elif variant in ("bf16", "bf16+ckpt"):
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = None

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    losses = []
    t0 = time.perf_counter()
    for step in range(total_steps):
        cond_image = torch.randn(batch_size, 3, 64, 64, device=device)
        x_0 = torch.randn(batch_size, model.horizon, model.action_dim, device=device)
        t = torch.randint(0, sched_noise.num_steps, (batch_size,), device=device)
        eps = torch.randn_like(x_0)
        x_t = sched_noise.q_sample(x_0, t, eps)

        opt.zero_grad(set_to_none=True)
        if amp_dtype is not None:
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                eps_hat = model(x_t, t, cond_image)
                loss = F.mse_loss(eps_hat, eps)
        else:
            eps_hat = model(x_t, t, cond_image)
            loss = F.mse_loss(eps_hat, eps)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        losses.append(loss.item())

    if use_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    throughput = total_steps / elapsed

    if use_cuda:
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        # Approximation CPU : on remonte le compte des parametres a la louche.
        # Sur CPU on n'a pas de measure peak fiable sans tracemalloc.
        peak_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)

    final_loss = float(np.mean(losses[-50:]))
    return throughput, peak_mb, final_loss


def solution_hard(out_dir: Path = Path("out_j26_hard"), total_steps: int = 200):
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = ["fp32", "fp16", "bf16", "bf16+ckpt"]
    results = {}
    for v in variants:
        try:
            tp, mem, loss = _bench_one(v, total_steps=total_steps)
            results[v] = (tp, mem, loss)
            print(f"{v:<12s}  throughput={tp:.1f} st/s  peak_mem={mem:.1f} MB  loss={loss:.4f}")
        except Exception as exc:
            print(f"{v:<12s}  FAILED : {exc}")
            results[v] = (None, None, None)

    # Tableau Markdown.
    table = "| Variante | Throughput (st/s) | Peak mem (MB) | Loss finale |\n"
    table += "|---|---|---|---|\n"
    for v in variants:
        tp, mem, loss = results[v]
        if tp is None:
            table += f"| {v} | FAILED | FAILED | FAILED |\n"
        else:
            table += f"| {v} | {tp:.1f} | {mem:.1f} | {loss:.4f} |\n"
    (out_dir / "benchmark.md").write_text(table, encoding="utf-8")
    print("\n" + table)
    return results


# ===========================================================================
# Entree script
# ===========================================================================


def main():
    print("=== Solution EASY ===")
    solution_easy()
    print("\n=== Solution MEDIUM ===")
    solution_medium()
    print("\n=== Solution HARD ===")
    solution_hard()


if __name__ == "__main__":
    main()
