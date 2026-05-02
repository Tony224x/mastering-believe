"""
J11 — Solutions des exercices : REINFORCE -> A2C/baseline -> GAE -> PPO continu.

# requires: torch, gymnasium

Trois fonctions main appellables independamment :
- solve_easy()    : REINFORCE vanilla sur CartPole-v1
- solve_medium()  : REINFORCE + baseline V(s) + GAE, comparaison
- solve_hard()    : esquisse du PPO continu sur HalfCheetah-v4 (squelette
                    pret a etre execute, necessite gymnasium[mujoco] installe)

Sources :
- [Schulman et al., 2017, PPO]
- [Schulman et al., 2016, GAE]
- [OpenAI Spinning Up, VPG / PPO]
- [CleanRL ppo_continuous_action.py]
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal


# ===========================================================================
# EASY — REINFORCE vanilla
# ===========================================================================
class PolicyMLP(nn.Module):
    """MLP simple qui sort des logits sur les actions discretes."""

    def __init__(self, obs_dim: int, n_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def returns_to_go(rewards: list[float], gamma: float) -> np.ndarray:
    """Calcule R_t = sum_{k>=t} gamma^{k-t} r_k en O(T)."""
    R = np.zeros(len(rewards), dtype=np.float32)
    G = 0.0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        R[t] = G
    return R


def solve_easy(seed: int = 42, n_episodes: int = 500) -> list[float]:
    """REINFORCE vanilla sur CartPole-v1.
    Reponses :
    - ~300-500 episodes pour atteindre retour moyen > 200 (3 seeds).
    - Avec R(tau) au lieu de return-to-go : meme esperance mais variance
      bien plus haute (chaque (s_t, a_t) est multiplie par TOUTE la
      recompense future ET passee, ce dernier terme etant pure bruit).
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    policy = PolicyMLP(obs_dim, n_actions)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    gamma = 0.99
    history: list[float] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        log_probs: list[torch.Tensor] = []
        rewards: list[float] = []
        done = False
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            logits = policy(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            obs, r, terminated, truncated, _ = env.step(int(action.item()))
            rewards.append(float(r))
            done = terminated or truncated

        R = returns_to_go(rewards, gamma)
        # Normalisation des returns : reduit la variance, optionnel mais
        # conventionnel (Spinning Up le fait par defaut).
        R = (R - R.mean()) / (R.std() + 1e-8)

        loss = -torch.stack([
            lp * torch.as_tensor(r, dtype=torch.float32)
            for lp, r in zip(log_probs, R)
        ]).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append(sum(rewards))
        if (ep + 1) % 50 == 0:
            mean = float(np.mean(history[-50:]))
            print(f"[easy] ep {ep + 1:4d}/{n_episodes}  mean_return(last 50)={mean:6.1f}")

    env.close()
    return history


# ===========================================================================
# MEDIUM — baseline V(s) + GAE, et comparaison
# ===========================================================================
class ActorCriticDiscrete(nn.Module):
    """Actor-critic discret : 2 heads sur un body partage."""

    def __init__(self, obs_dim: int, n_actions: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )
        self.actor = nn.Linear(64, n_actions)
        self.critic = nn.Linear(64, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(obs)
        return self.actor(h), self.critic(h).squeeze(-1)


def gae_episode(
    rewards: np.ndarray, values: np.ndarray, gamma: float, lam: float
) -> tuple[np.ndarray, np.ndarray]:
    """GAE pour un seul episode complet (terminal a la fin -> next_value = 0)."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_v = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_v - values[t]
        last_gae = delta + gamma * lam * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def solve_medium(
    seed: int = 42, n_episodes: int = 300, mode: str = "gae"
) -> list[float]:
    """REINFORCE + baseline ou GAE.
    mode : 'baseline' (R_t - V(s_t)) ou 'gae' (advantage GAE).

    Apres comparaison sur 5 seeds : GAE > baseline > REINFORCE en convergence.
    Pourquoi `lambda < 1` : `lambda = 1` retombe sur Monte Carlo (low bias /
    high variance), `lambda = 0` retombe sur 1-step TD (high bias / low var).
    `lambda = 0.95` est le sweet spot empirique sur la plupart des taches RL.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = ActorCriticDiscrete(obs_dim, n_actions)
    optimizer = optim.Adam(agent.parameters(), lr=1e-3)

    gamma, lam = 0.99, 0.95
    history: list[float] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        obs_list, action_list, reward_list, value_list, logp_list = [], [], [], [], []
        done = False
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            logits, value = agent(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()

            obs_list.append(obs.copy())
            action_list.append(int(action.item()))
            value_list.append(float(value.item()))
            logp_list.append(dist.log_prob(action))

            obs, r, terminated, truncated, _ = env.step(int(action.item()))
            reward_list.append(float(r))
            done = terminated or truncated

        rewards = np.array(reward_list, dtype=np.float32)
        values = np.array(value_list, dtype=np.float32)

        if mode == "baseline":
            R = returns_to_go(reward_list, gamma)
            advantages = R - values
            returns = R
        elif mode == "gae":
            advantages, returns = gae_episode(rewards, values, gamma, lam)
        else:
            raise ValueError(f"unknown mode {mode!r}")

        # Normalisation pour reduire la variance des updates.
        adv_t = torch.as_tensor(advantages, dtype=torch.float32)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.as_tensor(returns, dtype=torch.float32)

        # Recompute values (sinon graphe perdu) — on re-forward.
        obs_batch = torch.as_tensor(np.stack(obs_list), dtype=torch.float32)
        logits_b, values_b = agent(obs_batch)
        dist_b = Categorical(logits=logits_b)
        actions_b = torch.as_tensor(action_list, dtype=torch.int64)
        new_logp = dist_b.log_prob(actions_b)

        pg_loss = -(new_logp * adv_t).mean()
        v_loss = ((values_b - ret_t) ** 2).mean()
        loss = pg_loss + 0.5 * v_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append(float(rewards.sum()))
        if (ep + 1) % 50 == 0:
            mean = float(np.mean(history[-50:]))
            print(f"[medium/{mode}] ep {ep + 1:4d}/{n_episodes}  mean_return(last 50)={mean:6.1f}")

    env.close()
    return history


# ===========================================================================
# HARD — PPO continu sur HalfCheetah-v4 (squelette CleanRL-like)
# ===========================================================================
@dataclass
class PPOConfig:
    env_id: str = "HalfCheetah-v4"
    seed: int = 1
    total_timesteps: int = 1_000_000
    n_envs: int = 4
    n_steps: int = 2048
    n_epochs: int = 10
    minibatch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0  # tendance a baisser sur action continue
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    anneal_lr: bool = True


class ActorCriticContinuous(nn.Module):
    """Policy continue : Normal(mu, sigma) avec sigma state-independent."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )
        # log_sigma libre, partage entre tous les etats. Init a 0 -> sigma=1.
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.actor_mean(obs)
        logstd = self.actor_logstd.expand_as(mu)
        std = logstd.exp()
        dist = Normal(mu, std)
        if action is None:
            action = dist.sample()
        # Joint log-prob = somme des log-probs sur les dim independantes.
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.get_value(obs)
        return action, log_prob, entropy, value


def solve_hard(cfg: PPOConfig | None = None) -> None:
    """PPO continu sur HalfCheetah-v4. Necessite gymnasium[mujoco].

    Attendu : retour moyen > 1500 a 1M steps avec hyperparametres standard.
    `clip_fraction` typiquement 0.1-0.3, `approx_kl` < 0.02.

    Reponses :
    - n_epochs = 1 : equivalent a un single gradient step on-policy, faible
      utilisation des donnees mais tres conservateur. Convergence lente.
    - n_epochs = 50 : on sur-exploite le batch, le ratio s'eloigne fortement
      de 1, le clip sature partout (clip_fraction -> 0.5+). La KL explose
      et la performance peut s'effondrer. C'est exactement le scenario que
      le clip est cense empecher mais qu'il ne peut pas guerir totalement
      si on insiste trop.
    """
    if cfg is None:
        cfg = PPOConfig()

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def make_env(idx: int):
        def thunk():
            env = gym.make(cfg.env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(
                env, lambda o: np.clip(o, -10, 10), env.observation_space
            )
            env = gym.wrappers.NormalizeReward(env, gamma=cfg.gamma)
            env = gym.wrappers.TransformReward(env, lambda r: float(np.clip(r, -10, 10)))
            env.reset(seed=cfg.seed + idx)
            return env
        return thunk

    envs = gym.vector.SyncVectorEnv([make_env(i) for i in range(cfg.n_envs)])
    obs_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]

    agent = ActorCriticContinuous(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)

    obs_buf = torch.zeros((cfg.n_steps, cfg.n_envs, obs_dim), device=device)
    actions_buf = torch.zeros((cfg.n_steps, cfg.n_envs, action_dim), device=device)
    logprobs_buf = torch.zeros((cfg.n_steps, cfg.n_envs), device=device)
    rewards_buf = torch.zeros((cfg.n_steps, cfg.n_envs), device=device)
    dones_buf = torch.zeros((cfg.n_steps, cfg.n_envs), device=device)
    values_buf = torch.zeros((cfg.n_steps, cfg.n_envs), device=device)

    obs_np, _ = envs.reset(seed=cfg.seed)
    next_obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    next_done = torch.zeros(cfg.n_envs, device=device)

    n_updates = cfg.total_timesteps // (cfg.n_steps * cfg.n_envs)
    t0 = time.time()
    global_step = 0

    for update in range(1, n_updates + 1):
        if cfg.anneal_lr:
            frac = 1.0 - (update - 1.0) / n_updates
            for g in optimizer.param_groups:
                g["lr"] = frac * cfg.lr

        # Rollout
        for t in range(cfg.n_steps):
            global_step += cfg.n_envs
            obs_buf[t] = next_obs
            dones_buf[t] = next_done
            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(next_obs)
            actions_buf[t] = action
            logprobs_buf[t] = log_prob
            values_buf[t] = value

            obs_np, r, term, trunc, info = envs.step(action.cpu().numpy())
            done = np.logical_or(term, trunc)
            rewards_buf[t] = torch.as_tensor(r, dtype=torch.float32, device=device)
            next_obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(done.astype(np.float32), device=device)

        # Bootstrap + GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs)
            advantages = torch.zeros_like(rewards_buf)
            last_gae = torch.zeros(cfg.n_envs, device=device)
            for t in reversed(range(cfg.n_steps)):
                if t == cfg.n_steps - 1:
                    next_non_terminal = 1.0 - next_done
                    next_v = next_value
                else:
                    next_non_terminal = 1.0 - dones_buf[t + 1]
                    next_v = values_buf[t + 1]
                delta = rewards_buf[t] + cfg.gamma * next_v * next_non_terminal - values_buf[t]
                last_gae = delta + cfg.gamma * cfg.gae_lambda * next_non_terminal * last_gae
                advantages[t] = last_gae
            returns = advantages + values_buf

        # Flatten (n_steps, n_envs, ...) -> (n_steps * n_envs, ...)
        b_obs = obs_buf.reshape(-1, obs_dim)
        b_actions = actions_buf.reshape(-1, action_dim)
        b_logp = logprobs_buf.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)

        b_size = cfg.n_steps * cfg.n_envs
        idx = np.arange(b_size)
        clip_fracs: list[float] = []
        for epoch in range(cfg.n_epochs):
            np.random.shuffle(idx)
            for start_mb in range(0, b_size, cfg.minibatch_size):
                mb = idx[start_mb : start_mb + cfg.minibatch_size]
                mb_t = torch.as_tensor(mb, dtype=torch.int64, device=device)

                _, new_logp, entropy, new_v = agent.get_action_and_value(
                    b_obs[mb_t], b_actions[mb_t]
                )
                logratio = new_logp - b_logp[mb_t]
                ratio = logratio.exp()

                with torch.no_grad():
                    # CleanRL approximation of KL : ((r-1) - log r).mean()
                    approx_kl = ((ratio - 1.0) - logratio).mean().item()
                    clip_fracs.append(
                        ((ratio - 1.0).abs() > cfg.clip_eps).float().mean().item()
                    )

                mb_adv = b_adv[mb_t]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg1 = ratio * mb_adv
                pg2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_adv
                pg_loss = -torch.min(pg1, pg2).mean()

                v_loss = 0.5 * ((new_v - b_ret[mb_t]) ** 2).mean()
                ent_loss = entropy.mean()
                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

        if update % 5 == 0 or update == 1:
            elapsed = time.time() - t0
            mean_clip = float(np.mean(clip_fracs)) if clip_fracs else 0.0
            print(
                f"[hard] update {update}/{n_updates} "
                f"step={global_step} approx_kl={approx_kl:.4f} "
                f"clip_frac={mean_clip:.3f} elapsed={elapsed:.1f}s"
            )

    envs.close()


# ===========================================================================
# Demo
# ===========================================================================
if __name__ == "__main__":
    print("=== EASY: REINFORCE vanilla on CartPole-v1 ===")
    solve_easy(seed=42, n_episodes=200)

    print("\n=== MEDIUM: baseline + GAE comparison ===")
    print("--- baseline ---")
    solve_medium(seed=42, n_episodes=150, mode="baseline")
    print("--- gae ---")
    solve_medium(seed=42, n_episodes=150, mode="gae")

    # Le hard est laisse comme template, decommente si gymnasium[mujoco] dispo
    # print("\n=== HARD: PPO continuous on HalfCheetah-v4 (long !) ===")
    # solve_hard()
