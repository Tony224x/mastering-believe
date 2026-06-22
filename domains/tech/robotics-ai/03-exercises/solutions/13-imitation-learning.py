"""
J13 - Solutions des exercices Imitation Learning.

# requires: torch, gymnasium, numpy

Trois solutions :
    - solution_easy()   : BC ablation sur n_demos
    - solution_medium() : DAgger ablation sur rollouts_per_iter
    - solution_hard()   : GAIL minimal sur CartPole

Lance le tout avec : python 13-imitation-learning.py
Ou un seul exo via la variable RUN ci-dessous.

Note CPU : GAIL est plus lourd que BC/DAgger ; comptez 1-3 minutes CPU pour
solution_hard(). BC + DAgger : quelques secondes.

Sources : [Zare et al., 2024], [CS285 L2], [CS224R L2 - Finn], [Ho & Ermon, 2016].
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cpu")

# Choisir lesquels des 3 exos lancer en main()
RUN = ("easy", "medium", "hard")


# =============================================================================
# Outils communs
# =============================================================================
def expert_action(obs: np.ndarray) -> int:
    _, _, theta, theta_dot = obs
    return 1 if (theta + 0.5 * theta_dot) > 0 else 0


@dataclass
class Dataset:
    obs: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)

    def add(self, s: np.ndarray, a: int) -> None:
        self.obs.append(np.asarray(s, dtype=np.float32))
        self.actions.append(int(a))

    def to_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(np.stack(self.obs)).float()
        y = torch.tensor(self.actions, dtype=torch.long)
        return x, y

    def __len__(self) -> int:
        return len(self.actions)


class PolicyMLP(nn.Module):
    def __init__(self, obs_dim: int = 4, n_actions: int = 2, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, obs_np: np.ndarray, deterministic: bool = True) -> int:
        obs_t = torch.from_numpy(np.asarray(obs_np, dtype=np.float32)).unsqueeze(0)
        logits = self.forward(obs_t)
        if deterministic:
            return int(torch.argmax(logits, dim=-1).item())
        probs = F.softmax(logits, dim=-1)
        return int(torch.distributions.Categorical(probs=probs).sample().item())

    def sample_with_logp(self, obs_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action et retourne log_prob (pour policy gradient)."""
        logits = self.forward(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a)


def collect_expert_demos(env: gym.Env, n_episodes: int, base_seed: int = 0) -> Dataset:
    dataset = Dataset()
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        done = False
        while not done:
            a = expert_action(obs)
            dataset.add(obs, a)
            obs, _, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
    return dataset


def train_supervised(
    policy: PolicyMLP,
    dataset: Dataset,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> float:
    x, y = dataset.to_tensors()
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    optim = torch.optim.Adam(policy.parameters(), lr=lr)
    final_loss = 0.0
    for _ in range(epochs):
        running = 0.0
        n_batches = 0
        for xb, yb in loader:
            logits = policy(xb)
            loss = F.cross_entropy(logits, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += loss.item()
            n_batches += 1
        final_loss = running / max(1, n_batches)
    return final_loss


def evaluate(policy: PolicyMLP, env: gym.Env, n_eval: int = 20, base_seed: int = 10_000) -> float:
    returns = []
    for ep in range(n_eval):
        obs, _ = env.reset(seed=base_seed + ep)
        done = False
        ret = 0.0
        while not done:
            a = policy.act(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(a)
            ret += float(r)
            done = terminated or truncated
        returns.append(ret)
    return float(np.mean(returns))


# =============================================================================
# Solution EASY : BC ablation sur n_demos
# =============================================================================
def solution_easy() -> None:
    print("\n" + "=" * 70)
    print("EASY - BC ablation sur n_demos")
    print("=" * 70)
    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")

    results = {}
    for n_demos in (1, 5, 50):
        torch.manual_seed(SEED)  # meme init du reseau pour comparer fairement
        np.random.seed(SEED)
        demos = collect_expert_demos(env, n_episodes=n_demos)
        policy = PolicyMLP().to(DEVICE)
        train_supervised(policy, demos, epochs=30)
        score = evaluate(policy, eval_env, n_eval=20)
        results[n_demos] = (len(demos), score)
        print(f"  n_demos={n_demos:3d}  transitions={len(demos):4d}  return={score:6.1f}")

    print("\nLecture :")
    print("  Plus on a de demos, plus la couverture des etats voisins du sentier")
    print("  expert s'enrichit -> distribution shift attenue -> regret O(T^2 epsilon)")
    print("  reduit en pratique (cf. Ross & Bagnell 2010, [CS285 L2]).")
    env.close()
    eval_env.close()


# =============================================================================
# Solution MEDIUM : DAgger ablation sur rollouts_per_iter
# =============================================================================
def collect_student_states_with_expert_labels(
    policy: PolicyMLP, env: gym.Env, n_episodes: int, base_seed: int
) -> Dataset:
    dataset = Dataset()
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        done = False
        while not done:
            a_expert = expert_action(obs)
            a_student = policy.act(obs, deterministic=True)
            dataset.add(obs, a_expert)
            obs, _, terminated, truncated, _ = env.step(a_student)
            done = terminated or truncated
    return dataset


def merge_datasets(a: Dataset, b: Dataset) -> Dataset:
    out = Dataset()
    out.obs = list(a.obs) + list(b.obs)
    out.actions = list(a.actions) + list(b.actions)
    return out


def solution_medium() -> None:
    print("\n" + "=" * 70)
    print("MEDIUM - DAgger ablation sur rollouts_per_iter")
    print("=" * 70)
    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")

    initial_demos = collect_expert_demos(env, n_episodes=2)
    n_iter = 5

    print(f"\nDataset initial : {len(initial_demos)} transitions issues de 2 demos.")
    print(f"DAgger : {n_iter} iterations.\n")

    for rpi in (1, 5, 20):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        policy = PolicyMLP().to(DEVICE)
        dataset = Dataset()
        dataset.obs = list(initial_demos.obs)
        dataset.actions = list(initial_demos.actions)

        train_supervised(policy, dataset, epochs=20)
        scores = [evaluate(policy, eval_env, n_eval=20)]
        sizes = [len(dataset)]

        for it in range(n_iter):
            base = 50_000 + (it + 1) * 1000 + rpi
            new = collect_student_states_with_expert_labels(policy, env, rpi, base_seed=base)
            dataset = merge_datasets(dataset, new)
            train_supervised(policy, dataset, epochs=20)
            scores.append(evaluate(policy, eval_env, n_eval=20))
            sizes.append(len(dataset))

        print(f"  rollouts_per_iter={rpi:2d} | sizes={sizes} | returns={[round(s,1) for s in scores]}")

    print("\nLecture :")
    print("  - rpi=1  : feedback le plus reactif mais peu de couverture par iter.")
    print("  - rpi=5  : compromis usuel, converge en 2-3 iters.")
    print("  - rpi=20 : datasets plus gros mais le student n'evolue plus entre rollouts,")
    print("             on accumule des etats redondants -> moins efficient en transitions.")
    env.close()
    eval_env.close()


# =============================================================================
# Solution HARD : GAIL minimal
# =============================================================================
class Discriminator(nn.Module):
    """D(s, a) -> [0, 1]. 1 = expert, 0 = student."""

    def __init__(self, obs_dim: int = 4, n_actions: int = 2, hidden: int = 64):
        super().__init__()
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(obs_dim + n_actions, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        a_onehot = F.one_hot(a, num_classes=self.n_actions).float()
        x = torch.cat([s, a_onehot], dim=-1)
        return self.net(x)  # logits ; sigmoid applique cote loss


def gail_reward(D: Discriminator, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Reward `r = -log(1 - sigma(logit))`. Clamp pour stabilite."""
    with torch.no_grad():
        logit = D(s, a).squeeze(-1)
        # log_p = log sigma(logit) ; reward = log sigma(logit) - log sigma(-logit) (forme alt)
        # On utilise -log(1 - p) = -log(sigma(-logit)) = softplus(logit)
        r = F.softplus(logit)  # >= 0, monotone croissant en logit
        return torch.clamp(r, max=10.0)


def collect_student_rollouts(
    policy: PolicyMLP, env: gym.Env, n_episodes: int, base_seed: int
):
    """Roll-out la policy (stochastic) et retourne (obs, actions, log_probs, ep_lengths)."""
    obs_buf, act_buf, logp_buf, ep_lens = [], [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        done = False
        steps = 0
        while not done:
            obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
            logits = policy(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            logp = dist.log_prob(a)
            obs_buf.append(obs.astype(np.float32))
            act_buf.append(int(a.item()))
            logp_buf.append(logp.squeeze(0))
            obs, _, terminated, truncated, _ = env.step(int(a.item()))
            done = terminated or truncated
            steps += 1
        ep_lens.append(steps)
    obs_t = torch.from_numpy(np.stack(obs_buf)).float()
    act_t = torch.tensor(act_buf, dtype=torch.long)
    logp_t = torch.stack(logp_buf)
    return obs_t, act_t, logp_t, ep_lens


def compute_returns_per_step(rewards: torch.Tensor, ep_lens: list[int], gamma: float = 0.99) -> torch.Tensor:
    """Return-to-go par episode (no GAE, REINFORCE simple)."""
    out = torch.zeros_like(rewards)
    idx = 0
    for L in ep_lens:
        running = 0.0
        for t in reversed(range(L)):
            running = float(rewards[idx + t].item()) + gamma * running
            out[idx + t] = running
        idx += L
    # Normalisation pour stabiliser le gradient
    return (out - out.mean()) / (out.std() + 1e-8)


def solution_hard() -> None:
    print("\n" + "=" * 70)
    print("HARD - GAIL minimal sur CartPole")
    print("=" * 70)

    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")

    # 1) Demos expert
    n_expert_demos = 20
    expert_ds = collect_expert_demos(env, n_episodes=n_expert_demos)
    s_E, a_E = expert_ds.to_tensors()
    print(f"  Demos expert : {n_expert_demos} episodes, {len(expert_ds)} transitions.")

    # 2) Modeles
    torch.manual_seed(SEED)
    policy = PolicyMLP().to(DEVICE)
    D = Discriminator().to(DEVICE)
    opt_pi = torch.optim.Adam(policy.parameters(), lr=3e-4)
    opt_D = torch.optim.Adam(D.parameters(), lr=1e-3)

    n_outer = 25
    rollouts_per_iter = 8
    d_epochs = 2  # peu d'epochs disc pour eviter qu'il sature

    eval_history = []
    print(f"\n  Boucle GAIL : {n_outer} outer iter, {rollouts_per_iter} rollouts/iter\n")

    for it in range(n_outer):
        # a) collect rollouts student
        s_S, a_S, logp_S, ep_lens = collect_student_rollouts(
            policy, env, rollouts_per_iter, base_seed=70_000 + it * 1000
        )

        # b) update discriminateur (BCE) sur batch mixant expert + student
        idx_E = torch.randperm(len(s_E))[: len(s_S)]  # echantillonner le meme nb que student
        s_E_b, a_E_b = s_E[idx_E], a_E[idx_E]
        d_loss_val = 0.0
        for _ in range(d_epochs):
            logits_E = D(s_E_b, a_E_b).squeeze(-1)
            logits_S = D(s_S, a_S).squeeze(-1)
            # Expert -> label 1 ; Student -> label 0
            loss_E = F.binary_cross_entropy_with_logits(logits_E, torch.ones_like(logits_E))
            loss_S = F.binary_cross_entropy_with_logits(logits_S, torch.zeros_like(logits_S))
            loss = loss_E + loss_S
            opt_D.zero_grad()
            loss.backward()
            opt_D.step()
            d_loss_val = float(loss.item())

        # c) policy gradient avec rewards issus de D
        rewards = gail_reward(D, s_S, a_S)  # (T,)
        returns = compute_returns_per_step(rewards, ep_lens)
        # REINFORCE : minimiser - E[log_pi * R]
        pg_loss = -(logp_S * returns.detach()).mean()
        opt_pi.zero_grad()
        pg_loss.backward()
        opt_pi.step()

        if it % 5 == 0 or it == n_outer - 1:
            score = evaluate(policy, eval_env, n_eval=10, base_seed=20_000 + it)
            eval_history.append(score)
            print(
                f"  iter {it:2d} | D_loss={d_loss_val:.3f} (cible ~1.39 = ne distingue plus) "
                f"| ep_len_moyen={np.mean(ep_lens):.1f} | eval_return={score:.1f}"
            )

    final_score = evaluate(policy, eval_env, n_eval=20)
    print(f"\n  GAIL final return moyen (20 eval ep) : {final_score:.1f}")

    # Comparaison BC sur memes demos
    bc_pol = PolicyMLP().to(DEVICE)
    train_supervised(bc_pol, expert_ds, epochs=30)
    bc_score = evaluate(bc_pol, eval_env, n_eval=20)
    print(f"  BC (memes 20 demos) return moyen     : {bc_score:.1f}")
    print("\n  Commentaire : sur CartPole BC suffit deja avec 20 demos. L'interet")
    print("  pedagogique de GAIL ici est d'observer la convergence du discriminateur")
    print("  (D_loss -> ~2*log(2)=1.39 = perdu) et la montee du return policy.")

    env.close()
    eval_env.close()


# =============================================================================
def main() -> None:
    if "easy" in RUN:
        solution_easy()
    if "medium" in RUN:
        solution_medium()
    if "hard" in RUN:
        solution_hard()


if __name__ == "__main__":
    main()
