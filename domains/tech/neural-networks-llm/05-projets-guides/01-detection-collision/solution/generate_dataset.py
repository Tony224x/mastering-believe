"""
Synthetic dataset generator for LogiSim near-collision detection.

We simulate 5000 MOVE events of which ~3% are near-collisions, with plausible
contextual features. Collision features have a biased distribution (more
same-fleet robots nearby, fewer external units detected, night shift more
frequent) but with enough overlap so the problem is not trivial.

Goal: look close enough to a real EOD log so that the lessons learned
(metrics, weighting, threshold) are transferable. Not realistic enough to
judge a production model, but enough to learn from.
"""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    "friendly_units_within_5m",      # int 0-8 — same-fleet robots nearby
    "external_units_detected_last_60s",  # int 0-6 — external-fleet units detected
    "time_since_last_own_fleet_detect_s",  # 0-600
    "target_confidence",             # 0-1
    "drone_marked_zone",             # 0/1 — zone marked by inventory drone
    "motion_mode",                   # 0=ordered, 1=reactive, 2=preemptive
    "partial_telemetry_index",       # 0-1 (1 = heavily degraded telemetry)
    "night_shift",                   # 0/1
    "minutes_in_shift",              # 0-240
]


def _sample_clean_move(rng: np.random.Generator) -> np.ndarray:
    """Nominal move: 'global' distribution with moderate contextual bias.

    We want to avoid perfect signals: every feature overlaps widely with the
    collision class. That is the real problem when features are only a proxy
    of the context (which is the case in reality).
    """
    return np.array([
        rng.integers(0, 6),                     # 0-5 same-fleet robots nearby
        rng.integers(0, 6),                     # 0-5 external units detected
        rng.uniform(0, 600),                    # any time
        rng.beta(4, 2),                         # rather high confidence
        float(rng.random() < 0.5),
        rng.choice([0, 1, 2], p=[0.6, 0.25, 0.15]),
        rng.beta(2, 4),                         # rather good telemetry
        float(rng.random() < 0.2),
        rng.uniform(0, 240),
    ], dtype=np.float32)


def _sample_collision(rng: np.random.Generator) -> np.ndarray:
    """Near-collision: same feature space, but shifted distribution.

    The model must capture a combination of several features — no single
    feature is enough. That is what makes the task hard and useful.
    """
    return np.array([
        rng.integers(1, 8),                     # typically more same-fleet robots nearby
        rng.integers(0, 4),                     # typically fewer external units detected
        rng.uniform(0, 400),                    # own unit seen more recently
        rng.beta(2, 4),                         # rather low confidence
        float(rng.random() < 0.25),
        rng.choice([0, 1, 2], p=[0.25, 0.3, 0.45]),  # preemptive motion more likely
        rng.beta(4, 2),                         # rather degraded telemetry
        float(rng.random() < 0.45),
        rng.uniform(0, 240),
    ], dtype=np.float32)


def load_dataset(n_samples: int = 5000, collision_rate: float = 0.03,
                 seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generates (X, y) with ~collision_rate positives.

    Note: we generate each class independently then shuffle, instead of
    drawing class then sample. Simpler, same result.
    """
    rng = np.random.default_rng(seed)
    n_pos = int(n_samples * collision_rate)
    n_neg = n_samples - n_pos

    X_neg = np.stack([_sample_clean_move(rng) for _ in range(n_neg)])
    X_pos = np.stack([_sample_collision(rng) for _ in range(n_pos)])
    y_neg = np.zeros(n_neg, dtype=np.int64)
    y_pos = np.ones(n_pos, dtype=np.int64)

    X = np.concatenate([X_neg, X_pos])
    y = np.concatenate([y_neg, y_pos])

    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def stratified_split(X: np.ndarray, y: np.ndarray,
                      ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
                      seed: int = 42):
    """Stratified split: guarantees the same prevalence in every split.

    Critical here: without stratification a split could end up with 0
    positives and evaluation becomes impossible.
    """
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    def _split_class(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(idx)
        a = int(n * ratios[0])
        b = int(n * (ratios[0] + ratios[1]))
        return idx[:a], idx[a:b], idx[b:]

    p_tr, p_va, p_te = _split_class(pos_idx)
    n_tr, n_va, n_te = _split_class(neg_idx)

    tr = np.concatenate([p_tr, n_tr])
    va = np.concatenate([p_va, n_va])
    te = np.concatenate([p_te, n_te])
    rng.shuffle(tr)
    rng.shuffle(va)
    rng.shuffle(te)

    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])


if __name__ == "__main__":
    X, y = load_dataset()
    print(f"Dataset : {X.shape}, positifs = {y.sum()} ({y.mean():.2%})")
    (Xtr, ytr), (Xva, yva), (Xte, yte) = stratified_split(X, y)
    print(f"Train : {Xtr.shape}, positifs = {ytr.sum()}")
    print(f"Val   : {Xva.shape}, positifs = {yva.sum()}")
    print(f"Test  : {Xte.shape}, positifs = {yte.sum()}")
    print("Features :", FEATURE_NAMES)
