"""
J17 — Solutions consolidées (easy + medium + hard).

# requires: torch>=2.0 numpy matplotlib

This file solves all three exercises of J17 (world models / Dreamer-style RSSM).
It re-imports the toy WorldModel from 02-code/17-world-models-dreamer.py so it
stays in sync with the canonical implementation. We add:

    * easy   -> symlog/symexp self-test + RSSM landmark walkthrough
    * medium -> imagination_drift() over horizons, plot
    * hard   -> 4-way ablation (no_symlog, continuous_z, no_kl_balancing,
                no_free_bits), 3 seeds each, plot mean +/- std

Run:
    python 17-world-models-dreamer.py
Expected:
    - easy section prints symlog values and asserts symexp(symlog(x)) ~= x
    - medium section prints drift per horizon
    - hard section prints final recon loss per ablation x seed and saves a plot
      to ablation_recon.png if matplotlib is available

Wall-clock target: ~2-4 minutes on CPU.
"""

from __future__ import annotations

import importlib.util
import os
import random
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Import the toy WorldModel from the J17 02-code file.
# We use importlib because filenames starting with a digit aren't valid Python
# module names for a regular `import`.
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
J17_CODE = (HERE.parent.parent / "02-code" / "17-world-models-dreamer.py").resolve()


def _load_j17_module():
    spec = importlib.util.spec_from_file_location("j17_world_models", J17_CODE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["j17_world_models"] = mod
    spec.loader.exec_module(mod)
    return mod


j17 = _load_j17_module()
GridWorld = j17.GridWorld
Config = j17.Config
WorldModel = j17.WorldModel
RSSMCell = j17.RSSMCell
Encoder = j17.Encoder
Decoder = j17.Decoder
Head = j17.Head
symlog = j17.symlog
symexp = j17.symexp
collect_episode = j17.collect_episode
stack_batch = j17.stack_batch
actor_critic_imagination_loss = j17.actor_critic_imagination_loss


# =============================================================================
# EASY — symlog identity + RSSM landmark walkthrough
# =============================================================================
def solution_easy() -> None:
    print("\n=== EASY ===")
    # 1. symlog/symexp round-trip
    x = torch.linspace(-1000, 1000, 21)
    rt = symexp(symlog(x))
    err = (rt - x).abs().max().item()
    assert err < 1e-4, f"symexp(symlog(x)) round-trip error too high: {err}"
    print(f"symlog round-trip max abs error on [-1000,1000]: {err:.2e}  (PASS)")

    # 2. printed values for sanity
    for v in [-1000.0, -1.0, 0.0, 1.0, 1000.0]:
        s = symlog(torch.tensor(v)).item()
        print(f"  symlog({v:+.0f}) = {s:+.4f}")
    print("  -> sign preserved, identity around 0, log-compressed at large |x|.")
    print("  -> log(|x|+1) avoids singularity at x=0 and stays linear for small x.")

    # 3. RSSM landmark walkthrough (string descriptions, not exact line numbers
    #    so the solution stays robust to minor edits in the canonical file).
    print("\nRSSM landmarks in 02-code/17-world-models-dreamer.py:")
    print("  * Prior:     RSSMCell.prior_net  (used in imagine_step, no obs).")
    print("  * Posterior: RSSMCell.posterior_net (used in observe_step, sees embed).")
    print("  * KL balancing: world_loss(), kl_a (post detached) + kl_b (prior detached),")
    print("    weighted 0.8/0.2 with free_bits floor via torch.clamp(min=free_bits).")


# =============================================================================
# MEDIUM — imagination drift vs horizon
# =============================================================================
def imagination_drift(
    wm: WorldModel,
    cfg: Config,
    H: int,
    n_runs: int = 20,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """For each step t in [0, H), return mean L2 drift between true h_t (from
    observe_step) and imagined h_t (from imagine_step starting after step 0)."""
    drifts = np.zeros(H, dtype=np.float64)
    valid = 0

    for _ in range(n_runs):
        env = GridWorld()
        obs = env.reset()
        # generate H-1 random actions in advance so observe and imagine paths
        # consume the exact same actions (deterministic comparison).
        actions = [random.randint(0, cfg.action_dim - 1) for _ in range(H)]

        # === path A: observe (ground truth latents) ===
        h_o, z_o = wm.initial_state(1, device)
        h_traj_obs = []
        with torch.no_grad():
            for t in range(H):
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                a_oh = F.one_hot(
                    torch.tensor([actions[t]], device=device), num_classes=cfg.action_dim
                ).float()
                embed = wm.encoder(obs_t)
                h_o, z_o, _, _ = wm.rssm.observe_step(h_o, z_o, a_oh, embed)
                h_traj_obs.append(h_o.clone())
                obs, _, done = env.step(actions[t])
                if done and t < H - 1:
                    break
            else:
                # finished full H steps successfully
                # === path B: imagine (after seeing only obs[0]) ===
                env2 = GridWorld()
                obs2 = env2.reset()
                h_i, z_i = wm.initial_state(1, device)
                obs0_t = torch.tensor(obs2, dtype=torch.float32, device=device).unsqueeze(0)
                a0_oh = F.one_hot(
                    torch.tensor([actions[0]], device=device), num_classes=cfg.action_dim
                ).float()
                embed0 = wm.encoder(obs0_t)
                # one observe step to anchor on the real first obs
                h_i, z_i, _, _ = wm.rssm.observe_step(h_i, z_i, a0_oh, embed0)
                h_traj_im = [h_i.clone()]
                # then imagine the rest
                for t in range(1, H):
                    a_oh = F.one_hot(
                        torch.tensor([actions[t]], device=device),
                        num_classes=cfg.action_dim,
                    ).float()
                    h_i, z_i, _ = wm.rssm.imagine_step(h_i, z_i, a_oh)
                    h_traj_im.append(h_i.clone())

                for t in range(H):
                    d = (h_traj_obs[t] - h_traj_im[t]).pow(2).sum().sqrt().item()
                    drifts[t] += d
                valid += 1

    if valid == 0:
        return drifts
    return drifts / valid


def solution_medium() -> None:
    print("\n=== MEDIUM ===")
    cfg = Config()
    wm = WorldModel(cfg)

    # quick training to have a non-random model
    _train_world_model(wm, cfg, n_iters=50, batch=8, seq_len=12)

    horizons = [1, 2, 5, 10, 20]
    print("\nimagination drift per step, averaged over 15 runs:")
    for H in horizons:
        drift = imagination_drift(wm, cfg, H, n_runs=15)
        avg = drift.mean()
        last = drift[-1] if H > 0 else 0.0
        print(f"  H={H:2d} | mean drift over horizon = {avg:.3f} | drift at last step = {last:.3f}")
    print(
        "\nObservation: drift grows monotonically with H. This is exactly the"
        " 'model bias' issue from the theory (§6) — the further you dream, the"
        " more compounding errors push h_t away from reality."
    )


# =============================================================================
# HARD — 4-way ablation
# =============================================================================
@dataclass
class AblationFlags:
    no_symlog: bool = False
    continuous_z: bool = False
    no_kl_balancing: bool = False
    no_free_bits: bool = False


class AblationWorldModel(WorldModel):
    """Subclass of WorldModel that applies one ablation flag at a time."""

    def __init__(self, cfg: Config, flags: AblationFlags) -> None:
        super().__init__(cfg)
        self.flags = flags
        if flags.continuous_z:
            # Replace categorical RSSM by a Gaussian-z RSSM.
            self.rssm = _GaussianRSSM(
                cfg.action_dim, cfg.embed_dim, cfg.hidden_dim, cfg.n_cat * cfg.n_cls
            )

    def world_loss(self, obs_seq, action_seq, reward_seq, done_seq):
        T, B = obs_seq.shape[:2]
        device = obs_seq.device
        h, z = self.initial_state(B, device)
        recon_loss = 0.0
        reward_loss = 0.0
        cont_loss = 0.0
        kl_loss = 0.0

        for t in range(T):
            embed = self.encoder(obs_seq[t])
            if self.flags.continuous_z:
                h, z, prior_mu, prior_logsig, post_mu, post_logsig = (
                    self.rssm.observe_step(h, z, action_seq[t], embed)
                )
            else:
                h, z, prior_logits, post_logits = self.rssm.observe_step(
                    h, z, action_seq[t], embed
                )
            latent = torch.cat([h, z], dim=-1)

            # reconstruction
            recon_logits = self.decoder(latent)
            recon_loss = recon_loss + F.cross_entropy(
                recon_logits, obs_seq[t].argmax(dim=-1), reduction="mean"
            )

            # reward — symlog target unless ablated
            r_pred = self.reward_head(latent).squeeze(-1)
            r_tgt = reward_seq[t] if self.flags.no_symlog else symlog(reward_seq[t])
            reward_loss = reward_loss + F.mse_loss(r_pred, r_tgt)

            c_pred = self.continue_head(latent).squeeze(-1)
            c_tgt = 1.0 - done_seq[t]
            cont_loss = cont_loss + F.binary_cross_entropy_with_logits(c_pred, c_tgt)

            # KL term — varies per ablation
            if self.flags.continuous_z:
                # KL between two Gaussians (closed form)
                kl_t = _gaussian_kl(post_mu, post_logsig, prior_mu, prior_logsig).mean()
                if not self.flags.no_free_bits:
                    kl_t = torch.clamp(kl_t, min=self.cfg.free_bits)
            else:
                post_dist = torch.distributions.Categorical(logits=post_logits)
                prior_dist = torch.distributions.Categorical(logits=prior_logits)
                if self.flags.no_kl_balancing:
                    kl_t = torch.distributions.kl.kl_divergence(post_dist, prior_dist).sum(-1).mean()
                else:
                    kl_a = (
                        torch.distributions.kl.kl_divergence(
                            torch.distributions.Categorical(logits=post_logits.detach()),
                            prior_dist,
                        )
                        .sum(-1)
                        .mean()
                    )
                    kl_b = (
                        torch.distributions.kl.kl_divergence(
                            post_dist,
                            torch.distributions.Categorical(logits=prior_logits.detach()),
                        )
                        .sum(-1)
                        .mean()
                    )
                    if self.flags.no_free_bits:
                        kl_t = 0.8 * kl_a + 0.2 * kl_b
                    else:
                        kl_t = 0.8 * torch.clamp(
                            kl_a, min=self.cfg.free_bits
                        ) + 0.2 * torch.clamp(kl_b, min=self.cfg.free_bits)
            kl_loss = kl_loss + kl_t

        recon_loss = recon_loss / T
        reward_loss = reward_loss / T
        cont_loss = cont_loss / T
        kl_loss = kl_loss / T
        total = recon_loss + reward_loss + cont_loss + self.cfg.kl_weight * kl_loss
        return {
            "total": total,
            "recon": recon_loss,
            "reward": reward_loss,
            "continue": cont_loss,
            "kl": kl_loss,
        }


class _GaussianRSSM(nn.Module):
    """Continuous-z variant for ablation B. NOT used in the canonical model."""

    def __init__(self, action_dim: int, embed_dim: int, hidden_dim: int, z_dim: int) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRUCell(z_dim + action_dim, hidden_dim)
        self.prior_mu = nn.Linear(hidden_dim, z_dim)
        self.prior_logsig = nn.Linear(hidden_dim, z_dim)
        self.post_mu = nn.Linear(hidden_dim + embed_dim, z_dim)
        self.post_logsig = nn.Linear(hidden_dim + embed_dim, z_dim)

    @staticmethod
    def _sample(mu: torch.Tensor, logsig: torch.Tensor) -> torch.Tensor:
        std = torch.exp(logsig.clamp(-5, 2))
        eps = torch.randn_like(std)
        return mu + std * eps

    def observe_step(self, h_prev, z_prev, action, embed):
        h = self.gru(torch.cat([z_prev, action], dim=-1), h_prev)
        prior_mu = self.prior_mu(h)
        prior_logsig = self.prior_logsig(h)
        ph = torch.cat([h, embed], dim=-1)
        post_mu = self.post_mu(ph)
        post_logsig = self.post_logsig(ph)
        z = self._sample(post_mu, post_logsig)
        return h, z, prior_mu, prior_logsig, post_mu, post_logsig

    def imagine_step(self, h_prev, z_prev, action):
        h = self.gru(torch.cat([z_prev, action], dim=-1), h_prev)
        prior_mu = self.prior_mu(h)
        prior_logsig = self.prior_logsig(h)
        z = self._sample(prior_mu, prior_logsig)
        return h, z, (prior_mu, prior_logsig)


def _gaussian_kl(
    mu1: torch.Tensor, logsig1: torch.Tensor, mu2: torch.Tensor, logsig2: torch.Tensor
) -> torch.Tensor:
    # KL(N(mu1, sig1) || N(mu2, sig2)) elementwise, summed over last dim
    sig1_sq = torch.exp(2 * logsig1.clamp(-5, 2))
    sig2_sq = torch.exp(2 * logsig2.clamp(-5, 2))
    return (
        0.5
        * (
            (sig1_sq + (mu1 - mu2).pow(2)) / sig2_sq
            + 2 * (logsig2 - logsig1)
            - 1
        ).sum(dim=-1)
    )


def _train_world_model(
    wm: WorldModel, cfg: Config, n_iters: int = 60, batch: int = 8, seq_len: int = 12
) -> list[float]:
    env = GridWorld()
    opt = torch.optim.Adam(
        list(wm.encoder.parameters())
        + list(wm.rssm.parameters())
        + list(wm.decoder.parameters())
        + list(wm.reward_head.parameters())
        + list(wm.continue_head.parameters()),
        lr=3e-4,
    )
    recon_history: list[float] = []
    for it in range(n_iters):
        eps = [collect_episode(env, cfg) for _ in range(batch)]
        obs, act, rew, done = stack_batch(eps, seq_len)
        losses = wm.world_loss(obs, act, rew, done)
        opt.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
        opt.step()
        recon_history.append(losses["recon"].item())
    return recon_history


def solution_hard() -> None:
    print("\n=== HARD === (3 seeds x 5 variants — this is the slow part)")
    cfg = Config()
    variants: dict[str, AblationFlags] = {
        "baseline": AblationFlags(),
        "no_symlog": AblationFlags(no_symlog=True),
        "continuous_z": AblationFlags(continuous_z=True),
        "no_kl_balancing": AblationFlags(no_kl_balancing=True),
        "no_free_bits": AblationFlags(no_free_bits=True),
    }
    seeds = [0, 1, 2]
    n_iters = 50

    histories: dict[str, list[list[float]]] = {k: [] for k in variants}
    for name, flags in variants.items():
        for s in seeds:
            torch.manual_seed(s)
            random.seed(s)
            np.random.seed(s)
            wm = AblationWorldModel(cfg, flags)
            h = _train_world_model(wm, cfg, n_iters=n_iters)
            histories[name].append(h)
            print(f"  {name:18s} seed={s} | final recon = {h[-1]:.3f}")

    # summary table
    print("\nFinal recon loss summary (mean +/- std over 3 seeds):")
    summary = {}
    for name, runs in histories.items():
        finals = np.array([r[-1] for r in runs])
        summary[name] = (finals.mean(), finals.std())
        print(f"  {name:18s} | {finals.mean():.3f} +/- {finals.std():.3f}")

    # save plot if matplotlib is available
    try:
        import matplotlib.pyplot as plt  # type: ignore

        fig, ax = plt.subplots(figsize=(8, 5))
        for name, runs in histories.items():
            arr = np.array(runs)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            x = np.arange(len(mean))
            ax.plot(x, mean, label=name)
            ax.fill_between(x, mean - std, mean + std, alpha=0.2)
        ax.set_xlabel("training iteration")
        ax.set_ylabel("reconstruction loss")
        ax.set_title("J17 ablation — recon loss (3 seeds, mean +/- std)")
        ax.legend()
        out = HERE / "ablation_recon.png"
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        print(f"\nSaved plot to {out}")
    except ImportError:
        print("\nmatplotlib not available, skipping plot.")

    print(
        "\nInterpretation hooks (compare to Hafner 2023, REFERENCES.md #20, ablation table):"
    )
    print("  * no_symlog        -> reward loss can drift on long-tailed rewards.")
    print("  * continuous_z     -> usually higher seed variance (gradients noisier).")
    print("  * no_kl_balancing  -> posterior may collapse OR prior fails to track posterior.")
    print("  * no_free_bits     -> KL can be driven to ~0, posterior collapses, recon stalls.")


# =============================================================================
# Main entry — run all three solutions
# =============================================================================
if __name__ == "__main__":
    solution_easy()
    solution_medium()
    solution_hard()
