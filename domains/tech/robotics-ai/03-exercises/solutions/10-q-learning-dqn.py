"""
Solutions des exercices J10 - Q-learning, DQN
==============================================

- easy   : TD-error a la main + epsilon-greedy reproductible.
- medium : ReplayBuffer testable + comparaison SARSA vs Q-learning sur Cliff Walking.
- hard   : Double DQN avec mesure du biais d'overestimation (squelette d'entrainement
           parametrable et fonctions d'analyse). Pour rester rapide en CI / py_compile,
           le `if __name__ == "__main__"` execute easy + medium uniquement, et expose
           le hard via une fonction reutilisable.

# requires: torch, gymnasium, numpy
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
# EASY - TD-error et epsilon-greedy
# ---------------------------------------------------------------------------

def solve_easy() -> None:
    """Reponse aux 3 questions de l'exercice easy + check epsilon-greedy."""
    print("=" * 60)
    print("[EASY] TD-error a la main")
    print("=" * 60)
    Q = np.array(
        [
            [0.5, 0.3],
            [0.0, 1.0],
        ]
    )
    s, a, r, s_next = 0, 0, 0.1, 1
    alpha, gamma = 0.5, 0.9

    # cible TD : r + gamma * max_a' Q(s', a')
    td_target = r + gamma * float(np.max(Q[s_next]))
    td_error = td_target - Q[s, a]
    new_q = Q[s, a] + alpha * td_error
    print(f"cible TD       = {td_target:.4f}   (attendu : 0.1 + 0.9 * 1.0 = 1.0)")
    print(f"TD-error       = {td_error:.4f}    (attendu : 1.0 - 0.5 = 0.5)")
    print(f"Q[s0, a0] mis a jour = {new_q:.4f}  (attendu : 0.5 + 0.5 * 0.5 = 0.75)")

    # epsilon-greedy
    print()
    print("[EASY] epsilon-greedy")
    rng = np.random.default_rng(42)
    q_values = np.array([0.1, 0.5, 0.2, 0.4])

    def epsilon_greedy(q: np.ndarray, eps: float, generator: np.random.Generator) -> int:
        # Exploration : tirage uniforme dans [0, len(q)).
        if generator.random() < eps:
            return int(generator.integers(0, len(q)))
        # Exploitation : argmax (numpy retourne le 1er index en cas d'egalite, OK).
        return int(np.argmax(q))

    # Verif epsilon=0.0 : doit toujours retourner action 1 (Q max).
    actions_eps0 = [epsilon_greedy(q_values, 0.0, rng) for _ in range(100)]
    assert all(a == 1 for a in actions_eps0), "epsilon=0 doit etre purement greedy"
    print("epsilon=0.0  : toutes les actions == argmax (OK)")

    # Verif epsilon=1.0 : distribution uniforme sur 10000 tirages.
    rng2 = np.random.default_rng(0)
    actions_eps1 = [epsilon_greedy(q_values, 1.0, rng2) for _ in range(10_000)]
    counts = np.bincount(actions_eps1, minlength=4) / 10_000
    print(f"epsilon=1.0  : frequences observees = {counts}  (attendu ~ [0.25]*4)")
    assert np.all(np.abs(counts - 0.25) < 0.05), "uniformite cassee"


# ---------------------------------------------------------------------------
# MEDIUM - ReplayBuffer + Cliff Walking SARSA vs Q-learning
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """FIFO buffer avec sample sans remise, reproductible si le RNG est seede."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        self.buffer.append(
            Transition(np.asarray(state), int(action), float(reward), np.asarray(next_state), bool(done))
        )

    def sample(self, batch_size: int):
        # random.sample : tirage sans remise. Si on veut reproductibilite, on peut passer
        # par un random.Random local plutot que le module global.
        batch = random.sample(self.buffer, batch_size)
        states = np.array([t.state for t in batch])
        actions = np.array([t.action for t in batch])
        rewards = np.array([t.reward for t in batch])
        next_states = np.array([t.next_state for t in batch])
        dones = np.array([t.done for t in batch])
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


def test_replay_buffer() -> None:
    """Tests obligatoires de l'exercice medium partie A."""
    buf = ReplayBuffer(capacity=100)
    for i in range(1000):
        buf.push(np.array([i, i]), i % 2, float(i), np.array([i + 1, i + 1]), False)
    # Capacite respectee.
    assert len(buf) == 100
    # Les 100 derniers states sont presents (states 900..999).
    states_in_buffer = [t.state[0] for t in buf.buffer]
    assert min(states_in_buffer) == 900 and max(states_in_buffer) == 999

    # Sample renvoie les bonnes shapes.
    random.seed(0)
    s, a, r, ns, d = buf.sample(32)
    assert s.shape == (32, 2) and a.shape == (32,) and r.shape == (32,)

    # Reproductibilite : deux runs avec meme seed produisent meme batch.
    random.seed(123)
    s1, *_ = buf.sample(8)
    random.seed(123)
    s2, *_ = buf.sample(8)
    assert np.array_equal(s1, s2)
    print("[MEDIUM/A] ReplayBuffer : tous les tests passent")


# Cliff Walking (Sutton & Barto, ch. 6.5)
ROWS, COLS = 4, 12
START = (3, 0)
GOAL = (3, 11)


def step_cliff(state: tuple[int, int], action: int) -> tuple[tuple[int, int], float, bool]:
    r, c = state
    if action == 0:
        r = max(r - 1, 0)
    elif action == 1:
        r = min(r + 1, ROWS - 1)
    elif action == 2:
        c = max(c - 1, 0)
    elif action == 3:
        c = min(c + 1, COLS - 1)
    new_state = (r, c)
    # Cliff : ligne 3, colonnes 1..10 -> reward -100 + teleport au start.
    if new_state[0] == 3 and 1 <= new_state[1] <= 10:
        return START, -100.0, False
    if new_state == GOAL:
        return new_state, 0.0, True
    return new_state, -1.0, False


def epsilon_greedy_action(Q: np.ndarray, state: tuple[int, int], epsilon: float, rng: random.Random) -> int:
    if rng.random() < epsilon:
        return rng.randrange(4)
    return int(np.argmax(Q[state[0], state[1]]))


def run_sarsa_vs_qlearning(n_episodes: int = 500, alpha: float = 0.5, gamma: float = 1.0,
                            epsilon: float = 0.1, seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[float], list[float]]:
    """Entraine SARSA et Q-learning sur Cliff Walking. Retourne les Q-tables et les retours."""
    rng = random.Random(seed)
    Q_sarsa = np.zeros((ROWS, COLS, 4))
    Q_qlearn = np.zeros((ROWS, COLS, 4))
    returns_sarsa: list[float] = []
    returns_qlearn: list[float] = []

    for _ in range(n_episodes):
        # SARSA
        s = START
        a = epsilon_greedy_action(Q_sarsa, s, epsilon, rng)
        total_s = 0.0
        for _step in range(500):
            s_next, r, done = step_cliff(s, a)
            total_s += r
            a_next = epsilon_greedy_action(Q_sarsa, s_next, epsilon, rng)
            target = r if done else r + gamma * Q_sarsa[s_next[0], s_next[1], a_next]
            Q_sarsa[s[0], s[1], a] += alpha * (target - Q_sarsa[s[0], s[1], a])
            s, a = s_next, a_next
            if done:
                break
        returns_sarsa.append(total_s)

        # Q-learning
        s = START
        total_q = 0.0
        for _step in range(500):
            a = epsilon_greedy_action(Q_qlearn, s, epsilon, rng)
            s_next, r, done = step_cliff(s, a)
            total_q += r
            target = r if done else r + gamma * float(np.max(Q_qlearn[s_next[0], s_next[1]]))
            Q_qlearn[s[0], s[1], a] += alpha * (target - Q_qlearn[s[0], s[1], a])
            s = s_next
            if done:
                break
        returns_qlearn.append(total_q)

    return Q_sarsa, Q_qlearn, returns_sarsa, returns_qlearn


def render_path(Q: np.ndarray, name: str) -> None:
    arrows = {0: "^", 1: "v", 2: "<", 3: ">"}
    print(f"\nPolicy greedy {name} :")
    for r in range(ROWS):
        row = ""
        for c in range(COLS):
            if (r, c) == START:
                row += " S "
            elif (r, c) == GOAL:
                row += " G "
            elif r == 3 and 1 <= c <= 10:
                row += " . "  # cliff
            else:
                row += f" {arrows[int(np.argmax(Q[r, c]))]} "
        print(row)


def solve_medium() -> None:
    print("=" * 60)
    print("[MEDIUM/A] Tests ReplayBuffer")
    print("=" * 60)
    test_replay_buffer()

    print()
    print("=" * 60)
    print("[MEDIUM/B] Cliff Walking : SARSA vs Q-learning")
    print("=" * 60)
    Q_s, Q_q, rs, rq = run_sarsa_vs_qlearning(n_episodes=500)
    print(f"SARSA      : retour moyen 100 derniers episodes = {np.mean(rs[-100:]):.1f}")
    print(f"Q-learning : retour moyen 100 derniers episodes = {np.mean(rq[-100:]):.1f}")
    render_path(Q_s, "SARSA")
    render_path(Q_q, "Q-learning")
    print(
        "\nLecture : Q-learning apprend la trajectoire optimale (longe le cliff, retour ~ -13 sur"
        " un greedy run) mais explore avec eps=0.1 donc tombe parfois -> retour moyen plus bas."
        " SARSA prend l'exploration en compte (on-policy) et apprend une route plus eloignee du cliff,"
        " donc retours plus stables."
    )


# ---------------------------------------------------------------------------
# HARD - Double DQN
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TorchReplayBuffer:
    """ReplayBuffer dont sample renvoie des tensors directement (Double DQN, plus performant)."""

    def __init__(self, capacity: int) -> None:
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, *args) -> None:
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.buffer, batch_size)
        s = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32, device=device)
        a = torch.tensor([t.action for t in batch], dtype=torch.int64, device=device)
        r = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device)
        ns = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32, device=device)
        d = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device)
        return s, a, r, ns, d

    def __len__(self) -> int:
        return len(self.buffer)


def train_dqn_double(
    env_id: str = "CartPole-v1",
    total_steps: int = 30_000,
    double: bool = True,
    seed: int = 0,
    log_q_states: bool = True,
):
    """Entraine un DQN (vanilla ou double) et logge les Q-values moyennes sur un set d'etats fixe.

    Defaults sur CartPole pour rester rapide. Pour LunarLander, l'utilisateur passera
    `env_id="LunarLander-v3"` et `total_steps >= 100_000`.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(env_id)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)  # type: ignore[attr-defined]

    q_net = QNetwork(obs_dim, n_actions).to(device)
    target_net = QNetwork(obs_dim, n_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    for p in target_net.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(q_net.parameters(), lr=2.5e-4)
    buffer = TorchReplayBuffer(capacity=50_000)

    state, _ = env.reset(seed=seed)
    episode_returns: list[float] = []
    q_log: list[float] = []  # Q-value moyenne predite sur le set fixe.
    fixed_states: torch.Tensor | None = None
    ep_ret = 0.0

    for step in range(total_steps):
        progress = min(step / (total_steps * 0.4), 1.0)
        epsilon = 1.0 + (0.05 - 1.0) * progress

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = int(torch.argmax(q_net(s_t), dim=1).item())

        next_state, reward, terminated, truncated, _ = env.step(action)
        ep_ret += float(reward)
        buffer.push(
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(terminated),
        )

        if step >= 1000 and len(buffer) >= 64:
            s_b, a_b, r_b, ns_b, d_b = buffer.sample(64, device)
            with torch.no_grad():
                if double:
                    # Double DQN : argmax via online net, evaluation via target net.
                    next_a = q_net(ns_b).argmax(dim=1, keepdim=True)
                    target_q = target_net(ns_b).gather(1, next_a).squeeze(1)
                else:
                    # Vanilla DQN.
                    target_q = target_net(ns_b).max(dim=1).values
                td_target = r_b + 0.99 * target_q * (1.0 - d_b)
            pred = q_net(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(pred, td_target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
            optimizer.step()

        if step > 0 and step % 500 == 0:
            target_net.load_state_dict(q_net.state_dict())

        # Constituer le set d'etats fixes une fois (apres 1000 transitions collectees).
        if log_q_states and fixed_states is None and len(buffer) >= 1000:
            sample_idx = random.sample(range(len(buffer)), min(1000, len(buffer)))
            fixed_states = torch.tensor(
                np.array([buffer.buffer[i].state for i in sample_idx]),
                dtype=torch.float32,
                device=device,
            )

        if log_q_states and fixed_states is not None and step % 1000 == 0:
            with torch.no_grad():
                # Q-value max moyennee : indicateur de la magnitude des estimations.
                q_log.append(float(q_net(fixed_states).max(dim=1).values.mean().item()))

        if terminated or truncated:
            episode_returns.append(ep_ret)
            ep_ret = 0.0
            state, _ = env.reset()
        else:
            state = next_state

    env.close()
    return q_net, episode_returns, q_log


def solve_hard_demo() -> None:
    """Mini-demo du hard sur CartPole (rapide). Pour LunarLander, voir docstring du fichier."""
    print("=" * 60)
    print("[HARD - demo] Vanilla DQN vs Double DQN sur CartPole-v1")
    print("=" * 60)
    _, ret_vanilla, q_vanilla = train_dqn_double(double=False, total_steps=15_000, seed=0)
    _, ret_double, q_double = train_dqn_double(double=True, total_steps=15_000, seed=0)
    print(f"Vanilla DQN  : retour moyen 20 derniers episodes = {np.mean(ret_vanilla[-20:]):.1f}")
    print(f"Double DQN   : retour moyen 20 derniers episodes = {np.mean(ret_double[-20:]):.1f}")
    if q_vanilla and q_double:
        print(f"Q-value max moyenne finale - Vanilla : {q_vanilla[-1]:.2f}")
        print(f"Q-value max moyenne finale - Double  : {q_double[-1]:.2f}")
        print(
            "Sur CartPole le biais est faible (rewards petits) ; sur LunarLander/Atari Vanilla DQN"
            " a typiquement des Q-values nettement plus elevees que Double DQN (signature de l'overestimation)."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    solve_easy()
    print()
    solve_medium()
    print()
    # La demo hard tourne en ~1 minute sur CPU. Decommente pour exercer.
    # solve_hard_demo()
    print(
        "\n[HARD] Pour la version complete sur LunarLander-v3 :\n"
        "  - pip install 'gymnasium[box2d]'\n"
        "  - appeler train_dqn_double(env_id='LunarLander-v3', total_steps=200_000, double=True)\n"
        "  - logger episode_returns + q_log, repeter sur 3 seeds, comparer Vanilla vs Double."
    )
