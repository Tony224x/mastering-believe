"""
Generateur de dataset synthetique pour la detection de quasi-collisions LogiSim.

On simule 5000 events MOVE dont ~3% sont des quasi-collisions, avec des features
contextuelles plausibles. Les features de collisions ont une distribution
biaisee (plus de robots de la meme flotte proches, moins d'unites externes detectees,
shift de nuit plus frequent) mais avec assez de chevauchement pour que le
probleme ne soit pas trivial.

Objectif : ressemble suffisamment a un log EOD reel pour que les lecons
tirees (metriques, ponderation, seuil) soient transferables. Ce n'est pas
assez realiste pour juger un modele en production, mais assez pour apprendre.
"""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    "friendly_units_within_5m",      # int 0-8 — robots de la meme flotte proches
    "external_units_detected_last_60s",  # int 0-6 — unites de flotte externe detectees
    "time_since_last_own_fleet_detect_s",  # 0-600
    "target_confidence",             # 0-1
    "drone_marked_zone",             # 0/1 — zone marquee par drone d'inventaire
    "motion_mode",                   # 0=ordered, 1=reactive, 2=preemptive
    "partial_telemetry_index",       # 0-1 (1 = telemetrie tres degradee)
    "night_shift",                   # 0/1
    "minutes_in_shift",              # 0-240
]


def _sample_clean_move(rng: np.random.Generator) -> np.ndarray:
    """Mouvement nominal : distribution 'globale' avec biais contextuel modere.

    On veut eviter les signaux parfaits : chaque feature chevauche largement
    avec la classe collision. C'est un vrai probleme si les features ne
    sont qu'un proxy du contexte (ce qui est le cas en realite).
    """
    return np.array([
        rng.integers(0, 6),                     # 0-5 robots de la meme flotte proches
        rng.integers(0, 6),                     # 0-5 unites externes detectees
        rng.uniform(0, 600),                    # temps quelconque
        rng.beta(4, 2),                         # confidence plutot haute
        float(rng.random() < 0.5),
        rng.choice([0, 1, 2], p=[0.6, 0.25, 0.15]),
        rng.beta(2, 4),                         # telemetrie plutot bonne
        float(rng.random() < 0.2),
        rng.uniform(0, 240),
    ], dtype=np.float32)


def _sample_collision(rng: np.random.Generator) -> np.ndarray:
    """Quasi-collision : meme espace feature, mais distribution decalee.

    Le modele doit capturer une combinaison de plusieurs features — aucune
    seule feature ne suffit. C'est ce qui rend la tache difficile et utile.
    """
    return np.array([
        rng.integers(1, 8),                     # typiquement plus de robots de la meme flotte proches
        rng.integers(0, 4),                     # typiquement moins d'unites externes detectees
        rng.uniform(0, 400),                    # unite propre vue plus recemment
        rng.beta(2, 4),                         # confidence plutot basse
        float(rng.random() < 0.25),
        rng.choice([0, 1, 2], p=[0.25, 0.3, 0.45]),  # mouvement preemptif plus probable
        rng.beta(4, 2),                         # telemetrie plutot degradee
        float(rng.random() < 0.45),
        rng.uniform(0, 240),
    ], dtype=np.float32)


def load_dataset(n_samples: int = 5000, collision_rate: float = 0.03,
                 seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Genere (X, y) avec ~collision_rate positifs.

    Note : on genere independamment les classes puis on shuffle, au lieu de
    tirer class puis sample. Plus simple, meme resultat.
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
    """Split stratifie : garantit la meme prevalence dans chaque split.

    Critique ici : sans stratification, un split peut se retrouver avec 0
    positifs et l'eval devient impossible.
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
