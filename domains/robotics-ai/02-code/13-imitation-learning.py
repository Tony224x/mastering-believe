"""
J13 - Imitation Learning : BC + DAgger sur CartPole-v1.

# requires: torch, gymnasium, numpy

Objectif pedagogique :
    1. Construire un "expert" simple sur CartPole (politique heuristique deterministe
       qui equilibre le poteau).
    2. Generer N demos expert -> dataset (s, a).
    3. Entrainer une policy student par Behavior Cloning (BC).
    4. Mesurer perf BC (souvent bonne sur CartPole car episodes courts, mais on va
       voir qu'avec peu de demos elle souffre du distribution shift).
    5. Implementer DAgger : roll-out student, relabel par expert, retrain.
    6. Comparer BC vs DAgger en moyenne sur plusieurs seeds.

Cours : domains/robotics-ai/01-theory/13-imitation-learning.md
Sources : [Zare et al., 2024], [CS285 L2 - Levine], [Ross et al., 2011 - DAgger].

Note CPU : tout tourne en quelques secondes sur CPU (CartPole, MLP 64x64,
peu de demos, episodes courts).
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

# ---------------------------------------------------------------------------
# Reproductibilite : on fixe les seeds. CartPole etant deterministe une fois
# l'env seede, on aura des resultats stables.
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")  # ce module est CPU-friendly volontairement.


# ---------------------------------------------------------------------------
# 1. Expert heuristique sur CartPole-v1.
#
# CartPole : observation = [x, x_dot, theta, theta_dot] (4 floats).
#            action      = 0 (gauche) ou 1 (droite).
#
# Heuristique classique : "pousser dans la direction ou le poteau penche", en
# tenant compte aussi de la vitesse angulaire pour anticiper. Cette regle simple
# tient l'episode complet (500 steps) la plupart du temps -> expert ~optimal.
#
# Pourquoi pas un PPO entraine ? Parce qu'un script CPU-friendly self-contained
# doit demarrer en quelques secondes. Cette heuristique fait office d'oracle
# parfait pour la pedagogie. Le pattern BC/DAgger reste identique avec un
# expert PPO (cf. domains/robotics-ai/02-code/11-policy-gradients-ppo.py si
# besoin de remplacer).
# ---------------------------------------------------------------------------
def expert_action(obs: np.ndarray) -> int:
    """Politique expert heuristique pour CartPole. Retourne 0 ou 1."""
    _, _, theta, theta_dot = obs
    # On pousse a droite (action=1) si le poteau penche/tombe vers la droite.
    # theta + 0.5 * theta_dot anticipe legerement.
    return 1 if (theta + 0.5 * theta_dot) > 0 else 0


# ---------------------------------------------------------------------------
# 2. Reseau policy student : MLP simple 4 -> 64 -> 64 -> 2 logits.
# ---------------------------------------------------------------------------
class PolicyMLP(nn.Module):
    """Policy stochastique categorical pour CartPole (2 actions)."""

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
        """Retourne les logits (pas softmax, on utilisera cross_entropy)."""
        return self.net(obs)

    @torch.no_grad()
    def act(self, obs_np: np.ndarray, deterministic: bool = True) -> int:
        """Selection d'action a partir d'une obs numpy. Deterministic = argmax."""
        obs_t = torch.from_numpy(np.asarray(obs_np, dtype=np.float32)).unsqueeze(0)
        logits = self.forward(obs_t)
        if deterministic:
            return int(torch.argmax(logits, dim=-1).item())
        probs = F.softmax(logits, dim=-1)
        return int(torch.distributions.Categorical(probs=probs).sample().item())


# ---------------------------------------------------------------------------
# 3. Generation de demos expert.
# ---------------------------------------------------------------------------
@dataclass
class Dataset:
    """Petit container pour le dataset (s, a)."""

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


def collect_expert_demos(env: gym.Env, n_episodes: int) -> Dataset:
    """Roll-out de l'expert sur n_episodes -> dataset (s, a)."""
    dataset = Dataset()
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=SEED + ep)
        done = False
        while not done:
            a = expert_action(obs)
            dataset.add(obs, a)
            obs, _, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
    return dataset


# ---------------------------------------------------------------------------
# 4. Entrainement supervise (BC).
# ---------------------------------------------------------------------------
def train_supervised(
    policy: PolicyMLP,
    dataset: Dataset,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    verbose: bool = False,
) -> float:
    """Entrainement standard cross-entropy. Retourne la loss finale."""
    x, y = dataset.to_tensors()
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    optim = torch.optim.Adam(policy.parameters(), lr=lr)

    final_loss = 0.0
    for epoch in range(epochs):
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
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d}  loss={final_loss:.4f}")
    return final_loss


# ---------------------------------------------------------------------------
# 5. Evaluation : moyenne du return sur n_eval_episodes.
# ---------------------------------------------------------------------------
def evaluate(policy: PolicyMLP, env: gym.Env, n_eval_episodes: int = 20) -> float:
    """Retourne le return moyen (= longueur d'episode pour CartPole)."""
    returns = []
    for ep in range(n_eval_episodes):
        # On utilise des seeds disjointes du train pour eviter de tricher.
        obs, _ = env.reset(seed=10_000 + ep)
        done = False
        ep_return = 0.0
        while not done:
            a = policy.act(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(a)
            ep_return += float(r)
            done = terminated or truncated
        returns.append(ep_return)
    return float(np.mean(returns))


# ---------------------------------------------------------------------------
# 6. DAgger.
#
# Algorithme (cf. cours, section 3.1) :
#   D_0 = demos expert
#   pour i = 1..N :
#     1. roll-out de la policy courante -> visite des etats hors-distribution
#     2. relabel : pour chaque etat visite, on demande l'action a l'expert
#     3. D_{i} = D_{i-1} U {(s_t, a*_t)}
#     4. retrain policy supervise sur D_i (chaude, on continue depuis poids actuels)
# ---------------------------------------------------------------------------
def collect_student_states_with_expert_labels(
    policy: PolicyMLP, env: gym.Env, n_episodes: int, base_seed: int
) -> Dataset:
    """Roll-out la policy student, mais on enregistre (s, a*=expert(s))."""
    dataset = Dataset()
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        done = False
        while not done:
            # IMPORTANT : action expert pour le label, action student pour la transition.
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


def run_dagger(
    initial_dataset: Dataset,
    env: gym.Env,
    n_iterations: int = 5,
    rollouts_per_iter: int = 5,
    epochs_per_iter: int = 20,
    eval_env: gym.Env | None = None,
    n_eval_episodes: int = 20,
) -> tuple[PolicyMLP, list[float]]:
    """Boucle DAgger. Retourne la policy finale + l'historique d'eval."""
    policy = PolicyMLP().to(DEVICE)
    dataset = initial_dataset

    # Initial BC
    train_supervised(policy, dataset, epochs=epochs_per_iter)
    history = []
    if eval_env is not None:
        history.append(evaluate(policy, eval_env, n_eval_episodes))

    for it in range(n_iterations):
        # 1+2. Collect student rollouts + expert labels
        seed_offset = 50_000 + it * rollouts_per_iter  # seeds disjointes
        new_data = collect_student_states_with_expert_labels(
            policy, env, rollouts_per_iter, base_seed=seed_offset
        )
        # 3. Merge
        dataset = merge_datasets(dataset, new_data)
        # 4. Retrain (warm-start : on garde les poids courants, c'est intentionnel
        #    et plus rapide que reentrainer from scratch ; cf. CS285 L2).
        train_supervised(policy, dataset, epochs=epochs_per_iter)
        if eval_env is not None:
            history.append(evaluate(policy, eval_env, n_eval_episodes))
    return policy, history


# ---------------------------------------------------------------------------
# 7. Main : compare BC (peu de demos) vs DAgger.
#
# On choisit volontairement PEU de demos (n_demos=3) pour exhiber la fragilite
# de BC face au distribution shift. Avec beaucoup de demos sur CartPole, BC
# atteint deja le score max -> on ne verrait pas la difference.
# ---------------------------------------------------------------------------
def main() -> None:
    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")

    print("=" * 70)
    print("J13 - Imitation Learning : BC vs DAgger sur CartPole-v1")
    print("=" * 70)

    # --- Eval expert pour ancrer le score max ---
    expert_returns = []
    for ep in range(20):
        obs, _ = eval_env.reset(seed=10_000 + ep)
        done = False
        ret = 0.0
        while not done:
            a = expert_action(obs)
            obs, r, terminated, truncated, _ = eval_env.step(a)
            ret += float(r)
            done = terminated or truncated
        expert_returns.append(ret)
    print(f"\n[Expert heuristique] return moyen = {np.mean(expert_returns):.1f}  (max=500)")

    # --- 1) Generer demos expert ---
    n_demos = 3  # volontairement peu pour exhiber le distribution shift
    print(f"\n[1] Collecte de {n_demos} demos expert ...")
    demos = collect_expert_demos(env, n_episodes=n_demos)
    print(f"    -> dataset taille = {len(demos)} transitions (s, a)")

    # --- 2) Entrainer BC pur ---
    print("\n[2] Entrainement Behavior Cloning (BC) ...")
    bc_policy = PolicyMLP().to(DEVICE)
    final_loss = train_supervised(bc_policy, demos, epochs=30, verbose=False)
    bc_score = evaluate(bc_policy, eval_env, n_eval_episodes=20)
    print(f"    final loss = {final_loss:.4f}")
    print(f"    [BC] return moyen = {bc_score:.1f}")

    # --- 3) Entrainer DAgger en partant du meme dataset initial ---
    print("\n[3] Entrainement DAgger (5 iterations, 5 rollouts/iter) ...")
    dagger_policy, history = run_dagger(
        initial_dataset=demos,
        env=env,
        n_iterations=5,
        rollouts_per_iter=5,
        epochs_per_iter=20,
        eval_env=eval_env,
        n_eval_episodes=20,
    )
    print(f"    progression eval : {[round(h, 1) for h in history]}")
    print(f"    [DAgger] return moyen final = {history[-1]:.1f}")

    # --- 4) Comparaison ---
    print("\n" + "=" * 70)
    print("RESULTATS")
    print("=" * 70)
    print(f"  Expert    : {np.mean(expert_returns):6.1f}")
    print(f"  BC pur    : {bc_score:6.1f}    (meme dataset {len(demos)} transitions)")
    print(f"  DAgger    : {history[-1]:6.1f}    (apres 5 iter d'enrichissement)")
    print()
    print("Lecture :")
    print("  - BC fait souvent decrocher des episodes apres une petite erreur ;")
    print("    son score moyen est instable selon le seed.")
    print("  - DAgger comble le distribution shift en relabel-ant les etats")
    print("    visites par sa propre policy, et converge vers l'expert.")
    print("  - Avec n_demos plus grand (ex: 50), BC rattraperait DAgger sur")
    print("    CartPole - mais le pattern reste vrai sur taches plus longues.")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
