"""
J10 - Q-learning, DQN
======================

Deux algorithmes complementaires :
1) Q-learning tabulaire sur un GridWorld 5x5 (sans reseau de neurones).
2) DQN sur CartPole-v1 avec replay buffer + target network (Mnih et al., 2015).

Le code tourne sur CPU en quelques minutes (CartPole converge vers 200+ en
~150-300 episodes). Aucune dependance GPU.

# requires: torch, gymnasium, numpy

Sources :
- [Sutton & Barto, 2018, ch. 6 (TD control)]
- [Mnih et al., 2015] - "Human-level control through deep reinforcement learning"
- [CleanRL, dqn.py] - https://github.com/vwxyzjn/cleanrl
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Partie 1 - Q-learning tabulaire sur GridWorld
# ---------------------------------------------------------------------------

# GridWorld 5x5 : agent part en (0,0), objectif en (4,4) avec reward +1.
# Une cellule "trou" en (2,2) avec reward -1 (et fin d'episode).
# Actions : 0=haut, 1=bas, 2=gauche, 3=droite. Reward -0.01 par step (cout du temps).

GRID_SIZE = 5
GOAL = (GRID_SIZE - 1, GRID_SIZE - 1)
HOLE = (2, 2)


def step_gridworld(state: tuple[int, int], action: int) -> tuple[tuple[int, int], float, bool]:
    """Transition deterministe du GridWorld. Retourne (next_state, reward, done)."""
    r, c = state
    # Application de l'action avec clipping aux bords (l'agent ne sort pas du grid).
    if action == 0:
        r = max(r - 1, 0)
    elif action == 1:
        r = min(r + 1, GRID_SIZE - 1)
    elif action == 2:
        c = max(c - 1, 0)
    elif action == 3:
        c = min(c + 1, GRID_SIZE - 1)
    next_state = (r, c)
    if next_state == GOAL:
        return next_state, 1.0, True
    if next_state == HOLE:
        return next_state, -1.0, True
    # Petit cout par step : encourage l'agent a finir vite plutot que tourner en rond.
    return next_state, -0.01, False


def q_learning_tabular(
    n_episodes: int = 500,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
    """Q-learning tabulaire (Watkins, 1989) sur le GridWorld defini ci-dessus.

    Q[r, c, a] est mis a jour par la regle TD off-policy :
        Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

    Exploration : epsilon-greedy avec decay lineaire de `epsilon_start` a `epsilon_end`.
    """
    rng = random.Random(seed)
    # Q-table initialisee a 0. Shape (GRID_SIZE, GRID_SIZE, 4 actions).
    Q = np.zeros((GRID_SIZE, GRID_SIZE, 4), dtype=np.float64)
    returns: list[float] = []

    for episode in range(n_episodes):
        # Decay lineaire de epsilon : exploration forte au debut, exploitation a la fin.
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * (episode / max(1, n_episodes - 1))
        state = (0, 0)
        total_reward = 0.0
        # Limite de steps pour eviter les boucles infinies au debut de l'apprentissage.
        for _ in range(200):
            # Politique epsilon-greedy : exploration vs exploitation.
            if rng.random() < epsilon:
                action = rng.randrange(4)
            else:
                action = int(np.argmax(Q[state[0], state[1]]))

            next_state, reward, done = step_gridworld(state, action)
            total_reward += reward

            # Cible TD off-policy : si done, pas de bootstrap (max_a' Q(s',a') = 0 par convention).
            if done:
                td_target = reward
            else:
                td_target = reward + gamma * np.max(Q[next_state[0], next_state[1]])

            # Mise a jour incrementale Q-learning.
            td_error = td_target - Q[state[0], state[1], action]
            Q[state[0], state[1], action] += alpha * td_error

            state = next_state
            if done:
                break
        returns.append(total_reward)

    # Diagnostic : moyenne des 50 derniers retours, devrait etre proche de l'optimum (~0.92).
    print(f"[Q-learning tabulaire] return moyen sur les 50 derniers episodes : {np.mean(returns[-50:]):.3f}")
    return Q


def render_policy(Q: np.ndarray) -> None:
    """Affiche la policy greedy extraite de Q comme un grid de fleches."""
    arrows = {0: "^", 1: "v", 2: "<", 3: ">"}
    print("\nPolicy greedy apprise (G=goal, H=hole) :")
    for r in range(GRID_SIZE):
        row = ""
        for c in range(GRID_SIZE):
            if (r, c) == GOAL:
                row += " G "
            elif (r, c) == HOLE:
                row += " H "
            else:
                a = int(np.argmax(Q[r, c]))
                row += f" {arrows[a]} "
        print(row)
    print()


# ---------------------------------------------------------------------------
# Partie 2 - DQN sur CartPole-v1
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    """Une transition observee dans l'environnement."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Buffer circulaire FIFO. Decorrelle les echantillons (Mnih 2015 astuce 1).

    Stocker en numpy plutot qu'en tensors : moins d'allocations cote GPU/CPU.
    Le sampling renvoie des tensors prets a etre consommes par le reseau.
    """

    def __init__(self, capacity: int) -> None:
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        # random.sample fait un tirage SANS remise dans le buffer : echantillons distincts.
        batch = random.sample(self.buffer, batch_size)
        states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32, device=device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.int64, device=device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device)
        next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32, device=device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    """MLP simple : (obs_dim) -> 128 -> 128 -> (n_actions).

    L'output est Q(s, .) pour toutes les actions simultanement, donc
    forward(s) retourne un vecteur de taille n_actions.
    """

    def __init__(self, obs_dim: int, n_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_dqn(
    env_id: str = "CartPole-v1",
    total_steps: int = 30_000,
    buffer_capacity: int = 50_000,
    batch_size: int = 64,
    gamma: float = 0.99,
    lr: float = 5e-4,
    learning_starts: int = 1_000,
    train_frequency: int = 1,
    target_update_frequency: int = 500,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_steps: int = 10_000,
    seed: int = 0,
) -> QNetwork:
    """Entraine un DQN sur l'environnement Gymnasium passe en argument.

    Hyperparametres : configuration standard CleanRL adaptee CartPole.
    Sur CPU, atteint typiquement un retour moyen >=200 en 1-3 minutes.
    """
    # Determinisme : seed les 3 RNG critiques (numpy, torch, random).
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(env_id)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)  # type: ignore[attr-defined]

    # Reseaux online et target. Synchronisation initiale : memes poids.
    q_net = QNetwork(obs_dim, n_actions).to(device)
    target_net = QNetwork(obs_dim, n_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    # Le target_net ne s'entraine pas (pas de gradients) : on gele explicitement.
    for p in target_net.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_capacity)

    # Boucle d'interaction. On compte en STEPS d'env, pas en episodes.
    state, _ = env.reset(seed=seed)
    episode_return = 0.0
    episode_returns: list[float] = []

    for step in range(total_steps):
        # Decay lineaire de epsilon entre epsilon_start et epsilon_end sur epsilon_decay_steps.
        progress = min(step / epsilon_decay_steps, 1.0)
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * progress

        # epsilon-greedy.
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                q_values = q_net(state_t)
                action = int(torch.argmax(q_values, dim=1).item())

        next_state, reward, terminated, truncated, _ = env.step(action)
        # CartPole : `terminated` = chute du pendule, `truncated` = limite max steps atteinte.
        # Pour la cible Q-learning, seul `terminated` compte (il n'y a vraiment pas de futur).
        done_for_bootstrap = terminated
        episode_return += float(reward)

        buffer.push(
            Transition(
                state=np.asarray(state, dtype=np.float32),
                action=int(action),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32),
                done=bool(done_for_bootstrap),
            )
        )

        # On apprend uniquement quand le buffer est suffisamment rempli (sinon overfit aux 1ers echantillons).
        if step >= learning_starts and step % train_frequency == 0 and len(buffer) >= batch_size:
            states_b, actions_b, rewards_b, next_states_b, dones_b = buffer.sample(batch_size, device)

            # Cible TD : r + gamma * max_a' Q_target(s', a'), nullifiee si done.
            with torch.no_grad():
                target_max = target_net(next_states_b).max(dim=1).values
                td_target = rewards_b + gamma * target_max * (1.0 - dones_b)

            # Q(s, a) pour les a effectivement pris : gather sur l'axe action.
            q_pred = q_net(states_b).gather(1, actions_b.unsqueeze(1)).squeeze(1)

            # Smooth L1 (Huber) plus stable que MSE en cas de TD-error large (pratique CleanRL).
            loss = F.smooth_l1_loss(q_pred, td_target)

            optimizer.zero_grad()
            loss.backward()
            # Clip de gradient pour eviter les updates explosifs au debut de l'apprentissage.
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
            optimizer.step()

        # Hard update du target network toutes les `target_update_frequency` steps.
        if step % target_update_frequency == 0 and step > 0:
            target_net.load_state_dict(q_net.state_dict())

        if terminated or truncated:
            episode_returns.append(episode_return)
            episode_return = 0.0
            state, _ = env.reset()
            # Logging concis : moyenne mobile sur 20 episodes.
            if len(episode_returns) % 20 == 0:
                window = episode_returns[-20:]
                print(
                    f"[DQN] step={step:>6d} episode={len(episode_returns):>4d} "
                    f"epsilon={epsilon:.2f} return_mean_20={np.mean(window):.1f}"
                )
        else:
            state = next_state

    env.close()
    if episode_returns:
        print(f"[DQN] return moyen final (20 derniers episodes) : {np.mean(episode_returns[-20:]):.1f}")
    return q_net


def evaluate_dqn(q_net: QNetwork, env_id: str = "CartPole-v1", n_episodes: int = 10, seed: int = 123) -> float:
    """Evalue une policy greedy (epsilon=0) en moyennant le retour sur n episodes."""
    env = gym.make(env_id)
    device = next(q_net.parameters()).device
    returns: list[float] = []
    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        done = False
        total = 0.0
        while not done:
            with torch.no_grad():
                s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = int(torch.argmax(q_net(s_t), dim=1).item())
            state, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = terminated or truncated
        returns.append(total)
    env.close()
    mean_return = float(np.mean(returns))
    print(f"[DQN eval] retour moyen sur {n_episodes} episodes greedy : {mean_return:.1f}")
    return mean_return


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Partie 1 : Q-learning tabulaire sur GridWorld 5x5")
    print("=" * 60)
    Q = q_learning_tabular(n_episodes=500, alpha=0.1, gamma=0.99, seed=0)
    render_policy(Q)

    print("=" * 60)
    print("Partie 2 : DQN sur CartPole-v1")
    print("=" * 60)
    # 30k steps suffisent largement pour CartPole. Sur CPU : ~1-3 minutes.
    q_net = train_dqn(total_steps=30_000, seed=0)
    evaluate_dqn(q_net, n_episodes=10)
