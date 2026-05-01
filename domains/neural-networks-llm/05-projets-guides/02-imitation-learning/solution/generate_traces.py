"""
Generateur de traces expert rule-based pour behavioral cloning.

L'expert suit des regles simples :
- enemy proche + ammo ok + health ok -> ENGAGE
- enemy proche + health bas -> WITHDRAW
- enemy loin + pas en cover -> TAKE_COVER
- enemy loin + en cover + depuis longtemps sans ordre -> MOVE_FORWARD
- besoin de reporter -> REPORT
- sinon -> HOLD

Intentionnellement on melange un peu de bruit dans les decisions (10%) pour
donner un peu de variance au dataset : sinon c'est trivialement apprenable.
"""
from __future__ import annotations

import numpy as np

STATE_DIM = 7
ACTIONS = {
    "MOVE_FORWARD": 0,
    "TAKE_COVER": 1,
    "ENGAGE": 2,
    "WITHDRAW": 3,
    "HOLD": 4,
    "REPORT": 5,
}
ACTION_NAMES = [name for name, _ in sorted(ACTIONS.items(), key=lambda kv: kv[1])]


def _expert_policy(state: np.ndarray, in_cover: bool, rng: np.random.Generator) -> int:
    pos_x, pos_y, enemy_distance, enemy_count, health, ammo, since_order = state
    noise = rng.random()

    if enemy_distance < 0.3 and health > 0.3 and ammo > 0.2:
        action = ACTIONS["ENGAGE"]
    elif enemy_distance < 0.3 and (health <= 0.3 or ammo <= 0.2):
        action = ACTIONS["WITHDRAW"]
    elif enemy_distance < 0.6 and not in_cover:
        action = ACTIONS["TAKE_COVER"]
    elif since_order > 0.7 and enemy_distance > 0.7:
        action = ACTIONS["MOVE_FORWARD"]
    elif since_order > 0.9:
        action = ACTIONS["REPORT"]
    else:
        action = ACTIONS["HOLD"]

    # 10% de bruit : choisir une autre action au hasard
    if noise < 0.10:
        action = int(rng.integers(0, len(ACTIONS)))
    return action


def generate_sequence(rng: np.random.Generator, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Retourne (states, actions) ou states est (length, 7) et actions (length,)."""
    states = np.zeros((length, STATE_DIM), dtype=np.float32)
    actions = np.zeros(length, dtype=np.int64)

    pos = rng.uniform(0, 1, size=2)
    enemy_dist = rng.uniform(0.3, 1.0)
    enemy_count = rng.uniform(0, 1)
    health = 1.0
    ammo = 1.0
    since_order = 0.0
    in_cover = False

    for t in range(length):
        state = np.array([pos[0], pos[1], enemy_dist, enemy_count, health, ammo, since_order],
                         dtype=np.float32)
        action = _expert_policy(state, in_cover, rng)
        states[t] = state
        actions[t] = action

        # Dynamique simplifiee
        if action == ACTIONS["MOVE_FORWARD"]:
            pos += rng.uniform(-0.05, 0.05, size=2)
            enemy_dist = max(0.0, enemy_dist - 0.03)
            in_cover = False
            since_order = 0.0
        elif action == ACTIONS["TAKE_COVER"]:
            in_cover = True
            since_order = 0.0
        elif action == ACTIONS["ENGAGE"]:
            ammo = max(0.0, ammo - 0.05)
            health = max(0.0, health - rng.uniform(0, 0.08))
            enemy_count = max(0.0, enemy_count - 0.1)
            since_order = 0.0
        elif action == ACTIONS["WITHDRAW"]:
            pos -= rng.uniform(0, 0.05, size=2)
            enemy_dist = min(1.0, enemy_dist + 0.05)
            since_order = 0.0
        elif action == ACTIONS["REPORT"]:
            since_order = 0.0
        else:  # HOLD
            since_order = min(1.0, since_order + 0.05)

    return states, actions


def generate_dataset(n_sequences: int = 1000, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    dataset = []
    for _ in range(n_sequences):
        length = int(rng.integers(20, 100))
        dataset.append(generate_sequence(rng, length))
    return dataset


if __name__ == "__main__":
    data = generate_dataset(n_sequences=5)
    for i, (states, actions) in enumerate(data):
        print(f"Seq {i} len={len(states)} first action={ACTION_NAMES[actions[0]]}")
