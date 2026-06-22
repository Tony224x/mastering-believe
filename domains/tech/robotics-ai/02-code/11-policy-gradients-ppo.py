"""
J11 — Policy gradients & PPO simplifie sur CartPole-v1.

# requires: torch, gymnasium

Demo CPU-only en moins de 2 minutes :
- Implementation pedagogique de PPO (clip objective + GAE + value baseline + entropie).
- ~400 lignes inspirees de CleanRL `ppo_continuous_action.py` adapte au cas discret.
- CartPole-v1 atteint typiquement >450 de retour moyen en 30-60 secondes (CPU, 1 env).

Lance simplement :
    python 11-policy-gradients-ppo.py

Sources :
- [Schulman et al., 2017, PPO]  arxiv 1707.06347
- [OpenAI Spinning Up, PPO]      spinningup.openai.com
- [CleanRL ppo_continuous_action.py]  github.com/vwxyzjn/cleanrl
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ---------------------------------------------------------------------------
# Hyperparametres : volontairement ajustes pour une convergence rapide CPU.
# Le tuning de reference HalfCheetah serait 1M steps / 2048 rollout / 8 envs.
# Ici on cible CartPole-v1 et une demo < 2 minutes.
# ---------------------------------------------------------------------------
@dataclass
class Config:
    env_id: str = "CartPole-v1"
    seed: int = 42

    # Boucle d'entrainement
    total_timesteps: int = 25_000  # ~30s CPU pour atteindre solve threshold
    n_steps: int = 512             # taille du rollout avant chaque update
    n_epochs: int = 4              # passes SGD sur le meme batch
    minibatch_size: int = 64

    # PPO core
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5           # coefficient sur la value loss
    ent_coef: float = 0.01         # coefficient sur l'entropie (exploration)
    max_grad_norm: float = 0.5     # clipping global du gradient

    # Optim
    lr: float = 2.5e-4

    # Logging
    log_every: int = 5             # log toutes les N updates


# ---------------------------------------------------------------------------
# Reseau acteur-critique : MLP partage minimal.
# Pour CartPole l'observation est un vecteur 4D et l'action est discrete (2).
# ---------------------------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64) -> None:
        super().__init__()
        # On garde des heads separees pour eviter que la MSE de la value
        # n'interfere avec le signal de la policy (pratique courante CleanRL).
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),  # logits sur les actions discretes
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),  # scalar V(s)
        )

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        # On squeeze la derniere dim pour avoir un tenseur 1D (un V par etat).
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retourne (action, log_prob, entropy, value).
        Si une action est passee (cas du recompute des log-probs lors des
        epochs PPO), on calcule sa log-prob et son entropie au lieu d'en
        echantillonner une nouvelle.
        """
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.get_value(obs)
        return action, log_prob, entropy, value


# ---------------------------------------------------------------------------
# Calcul de l'advantage par GAE (Generalized Advantage Estimation).
# Formule : A_t = delta_t + (gamma * lambda) * A_{t+1}
# ou       delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t).
# Implementation backward pass : on parcourt le rollout en sens inverse.
# ---------------------------------------------------------------------------
def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    rewards / values / dones : arrays de shape (T,).
    next_value : V(s_T) — valeur du dernier etat suivant le rollout.
    Retourne (advantages, returns) ou returns = advantages + values.
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        # Si done_t == 1, le futur est tronque : on n'inclut pas V(s_{t+1}).
        if t == T - 1:
            next_non_terminal = 1.0 - dones[t]
            next_v = next_value
        else:
            next_non_terminal = 1.0 - dones[t]
            next_v = values[t + 1]
        delta = rewards[t] + gamma * next_v * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# Boucle principale.
# ---------------------------------------------------------------------------
def train(cfg: Config) -> None:
    # Reproductibilite
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Pour la demo on reste sur CPU : CartPole + petit MLP, GPU n'apporte rien.
    device = torch.device("cpu")

    env = gym.make(cfg.env_id)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = ActorCritic(obs_dim, n_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)

    # Buffers du rollout — on prealloue pour eviter les append() en chaud.
    obs_buf = np.zeros((cfg.n_steps, obs_dim), dtype=np.float32)
    actions_buf = np.zeros(cfg.n_steps, dtype=np.int64)
    logprobs_buf = np.zeros(cfg.n_steps, dtype=np.float32)
    rewards_buf = np.zeros(cfg.n_steps, dtype=np.float32)
    dones_buf = np.zeros(cfg.n_steps, dtype=np.float32)
    values_buf = np.zeros(cfg.n_steps, dtype=np.float32)

    obs, _ = env.reset(seed=cfg.seed)
    done_flag = 0.0
    episode_returns: list[float] = []
    current_episode_return = 0.0

    n_updates = cfg.total_timesteps // cfg.n_steps
    start = time.time()

    for update in range(1, n_updates + 1):
        # ----- 1. Rollout : on collecte n_steps transitions sous pi_old -----
        for t in range(cfg.n_steps):
            obs_buf[t] = obs
            dones_buf[t] = done_flag

            # Pas de gradient pendant le rollout : l'agent est juste en mode "actor".
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                action, log_prob, _, value = agent.get_action_and_value(obs_t)

            a_int = int(action.item())
            actions_buf[t] = a_int
            logprobs_buf[t] = float(log_prob.item())
            values_buf[t] = float(value.item())

            obs, reward, terminated, truncated, _ = env.step(a_int)
            rewards_buf[t] = float(reward)
            current_episode_return += float(reward)
            done_flag = 1.0 if (terminated or truncated) else 0.0

            if done_flag:
                episode_returns.append(current_episode_return)
                current_episode_return = 0.0
                obs, _ = env.reset()

        # Bootstrap : V(s_T) pour fermer l'estimation GAE proprement.
        with torch.no_grad():
            next_value = float(
                agent.get_value(
                    torch.as_tensor(obs, dtype=torch.float32, device=device)
                ).item()
            )

        # ----- 2. Compute advantages + returns -----
        advantages, returns = compute_gae(
            rewards_buf, values_buf, dones_buf, next_value,
            cfg.gamma, cfg.gae_lambda,
        )

        # ----- 3. Conversion en tenseurs pour les K epochs SGD -----
        b_obs = torch.as_tensor(obs_buf, dtype=torch.float32, device=device)
        b_actions = torch.as_tensor(actions_buf, dtype=torch.int64, device=device)
        b_logprobs_old = torch.as_tensor(logprobs_buf, dtype=torch.float32, device=device)
        b_advantages = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        b_returns = torch.as_tensor(returns, dtype=torch.float32, device=device)
        b_values_old = torch.as_tensor(values_buf, dtype=torch.float32, device=device)

        # ----- 4. PPO update : K epochs sur le meme batch, mini-batches shuffles -----
        idx = np.arange(cfg.n_steps)
        for epoch in range(cfg.n_epochs):
            np.random.shuffle(idx)
            for start_mb in range(0, cfg.n_steps, cfg.minibatch_size):
                mb = idx[start_mb : start_mb + cfg.minibatch_size]
                mb_t = torch.as_tensor(mb, dtype=torch.int64, device=device)

                _, new_logprobs, entropy, new_values = agent.get_action_and_value(
                    b_obs[mb_t], b_actions[mb_t]
                )

                # ratio r_t = pi_new(a|s) / pi_old(a|s) = exp(log_new - log_old)
                logratio = new_logprobs - b_logprobs_old[mb_t]
                ratio = logratio.exp()

                # Normalisation des advantages mini-batch (best-practice CleanRL).
                mb_adv = b_advantages[mb_t]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Clip objective : on prend le min entre objectif non-clipe et clipe.
                # Le sign moins vient du fait qu'on minimise -L^CLIP.
                pg_loss_unclipped = ratio * mb_adv
                pg_loss_clipped = torch.clamp(
                    ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps
                ) * mb_adv
                pg_loss = -torch.min(pg_loss_unclipped, pg_loss_clipped).mean()

                # Value loss : MSE entre V_phi(s) et returns observes.
                v_loss = 0.5 * ((new_values - b_returns[mb_t]) ** 2).mean()

                # Entropie : encourager la diversite des actions (exploration).
                ent_loss = entropy.mean()

                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent_loss

                optimizer.zero_grad()
                loss.backward()
                # Clipping global du gradient — empeche les pas extremes.
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

        # ----- 5. Logging compact -----
        if update % cfg.log_every == 0 or update == 1:
            recent = episode_returns[-20:] if episode_returns else [0.0]
            elapsed = time.time() - start
            steps = update * cfg.n_steps
            print(
                f"[update {update:3d}/{n_updates}] "
                f"steps={steps:6d} "
                f"mean_return={np.mean(recent):6.1f} "
                f"n_episodes={len(episode_returns):3d} "
                f"elapsed={elapsed:5.1f}s"
            )

    env.close()
    final_recent = episode_returns[-20:] if episode_returns else [0.0]
    print(f"\nDone. Final mean return (last 20 ep): {np.mean(final_recent):.1f}")
    print("CartPole-v1 'solved' threshold ~ 475. PPO converge typiquement >450 ici.")


if __name__ == "__main__":
    train(Config())
