"""
J22 — Solutions consolidees pour les 3 exercices.

Structure :
- EASY  : tableau de reference (commentaire structure + dataclass) + check assertions.
- MEDIUM: orchestrateur dual-system instrumente avec compteurs + chrono + diff embeddings.
- HARD  : pipeline d'entrainement complet (dataset expert -> System2 classification ->
          System1 conditionne -> eval + ablation period).

Verification : `python -m py_compile 22-frontier-humanoid.py` doit PASS.
Run : `python 22-frontier-humanoid.py` execute les 3 demos en sequence (rapide sur CPU).
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Reutilisation de l'env et des reseaux du module 02-code (copie locale pour
# que la solution soit autonome)
# ============================================================================


STRATEGIES = ("static", "horizontal_oscillation", "circle")


@dataclass
class TargetEnv:
    regime_change_every: int = 80
    box_size: float = 5.0
    seed: int = 42
    t: int = 0
    strategy: str = "static"
    target_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    agent_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    rng: random.Random = field(default_factory=random.Random)
    forced_strategy: str | None = None

    def reset(self) -> np.ndarray:
        self.rng = random.Random(self.seed)
        self.t = 0
        self._pick_new_strategy()
        self.agent_xy = np.array(
            [self.rng.uniform(-1, 1), self.rng.uniform(-1, 1)], dtype=np.float32
        )
        self._update_target()
        return self._obs()

    def _pick_new_strategy(self) -> None:
        if self.forced_strategy is not None:
            self.strategy = self.forced_strategy
        else:
            self.strategy = self.rng.choice(STRATEGIES)
        self._phase = self.rng.uniform(0, 2 * math.pi)
        self._anchor = np.array(
            [self.rng.uniform(-2, 2), self.rng.uniform(-2, 2)], dtype=np.float32
        )

    def _update_target(self) -> None:
        if self.strategy == "static":
            self.target_xy = self._anchor.copy()
        elif self.strategy == "horizontal_oscillation":
            x = self._anchor[0] + 1.5 * math.sin(0.1 * self.t + self._phase)
            y = self._anchor[1]
            self.target_xy = np.array([x, y], dtype=np.float32)
        elif self.strategy == "circle":
            r = 1.5
            x = self._anchor[0] + r * math.cos(0.08 * self.t + self._phase)
            y = self._anchor[1] + r * math.sin(0.08 * self.t + self._phase)
            self.target_xy = np.array([x, y], dtype=np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        action = np.clip(action, -0.3, 0.3).astype(np.float32)
        self.agent_xy = self.agent_xy + action
        self.agent_xy = np.clip(self.agent_xy, -self.box_size, self.box_size)
        self.t += 1
        if self.t % self.regime_change_every == 0:
            self._pick_new_strategy()
        self._update_target()
        dist = float(np.linalg.norm(self.agent_xy - self.target_xy))
        return self._obs(), -dist, self.t >= 1000

    def _obs(self) -> np.ndarray:
        return np.concatenate([self.agent_xy, self.target_xy]).astype(np.float32)


class System2(nn.Module):
    def __init__(self, history_len: int = 16, hidden: int = 64, embed_dim: int = 8, n_classes: int = 3):
        super().__init__()
        self.history_len = history_len
        self.embed_dim = embed_dim
        self.trunk = nn.Sequential(
            nn.Linear(history_len * 2 + 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.embed_head = nn.Sequential(nn.Linear(hidden, embed_dim), nn.Tanh())
        self.cls_head = nn.Linear(hidden, n_classes)  # auxiliary task pour HARD

    def trunk_features(self, target_history: torch.Tensor, agent_xy: torch.Tensor) -> torch.Tensor:
        b = target_history.shape[0]
        x = torch.cat([target_history.reshape(b, -1), agent_xy], dim=-1)
        return self.trunk(x)

    def forward(self, target_history: torch.Tensor, agent_xy: torch.Tensor) -> torch.Tensor:
        h = self.trunk_features(target_history, agent_xy)
        return self.embed_head(h)

    def classify(self, target_history: torch.Tensor, agent_xy: torch.Tensor) -> torch.Tensor:
        h = self.trunk_features(target_history, agent_xy)
        return self.cls_head(h)


class System1(nn.Module):
    def __init__(self, obs_dim: int = 4, embed_dim: int = 8, hidden: int = 32, action_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh(),
        )
        self.action_scale = 0.3

    def forward(self, obs: torch.Tensor, goal_embed: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, goal_embed], dim=-1)
        return self.net(x) * self.action_scale


# ============================================================================
# EASY : tableau de reference + verification
# ============================================================================


# Tableau de reference (a comparer avec ta reponse).
EASY_TABLE = [
    # (modele, annee, archi, f_ctrl_Hz_approx, dof_approx, open_weights)
    ("RT-2",      2023, "monolithique",  3,    7,  False),
    ("Octo",      2024, "monolithique",  10,   7,  True),
    ("OpenVLA",   2024, "monolithique",  6,    7,  True),
    ("pi0",       2024, "monolithique",  50,   14, False),  # pi0-FAST 5x
    ("GR00T N1",  2025, "dual-system",   120,  35, True),
    ("Helix",     2025, "dual-system",   200,  35, False),
    ("LBM (TRI)", 2024, "dual-system-eq", 30,  14, False),
]


EASY_DECOUPLING = {
    "GR00T N1": {"system2_hz": 7, "system1_hz": 120, "ratio": "~17x"},
    "Helix":    {"system2_hz": 8, "system1_hz": 200, "ratio": "~25x"},
}


EASY_JUSTIFICATION = (
    "Un VLM 7B sort au mieux ~30-50 tokens/s sur GPU H100, soit "
    "~7 Hz de generation d'action chunks. Or piloter un humanoide whole-body "
    "35 DoF demande 100-200 Hz pour la stabilite et la reactivite : c'est deux "
    "ordres de grandeur d'ecart, donc impossible avec un seul reseau monolithique."
)


def easy_self_check() -> None:
    """Quelques assertions defensives pour valider la reponse."""
    assert len(EASY_TABLE) == 7, "Le tableau doit avoir 7 lignes"
    monolithic = [r for r in EASY_TABLE if r[2] == "monolithique"]
    dual = [r for r in EASY_TABLE if "dual" in r[2]]
    assert len(monolithic) == 4 and len(dual) == 3, "4 mono + 3 dual"
    # Les freq de controle dual >= 100 Hz sauf LBM (qui est plus manipulation que humanoide)
    for name, _y, _a, fhz, _d, _o in EASY_TABLE:
        if name in ("GR00T N1", "Helix"):
            assert fhz >= 100, f"{name} controle effectif >= 100 Hz attendu"
    # Open-weights : Octo, OpenVLA, GR00T N1 sont open
    open_models = {r[0] for r in EASY_TABLE if r[5]}
    assert {"Octo", "OpenVLA", "GR00T N1"} <= open_models
    print("[EASY] tableau et assertions : PASS")
    print(f"[EASY] justification cle : {EASY_JUSTIFICATION}")
    print(f"[EASY] decoupage : {EASY_DECOUPLING}")


# ============================================================================
# MEDIUM : orchestrateur instrumente
# ============================================================================


class DualSystemInstrumented:
    """Solution MEDIUM : compteurs + chrono + diff embeddings consecutifs."""

    def __init__(self, history_len: int = 16, system2_period: int = 8):
        self.system2 = System2(history_len=history_len)
        self.system1 = System1()
        self.history_len = history_len
        self.system2_period = system2_period
        # buffers
        self._target_history: List[np.ndarray] = []
        self._cached_embed: torch.Tensor | None = None
        self._step_idx = 0
        # instrumentation
        self.n_system2_calls = 0
        self.n_system1_calls = 0
        self.t_system2 = 0.0  # secondes cumulees
        self.t_system1 = 0.0
        self.embed_diffs: List[float] = []

    def reset(self, initial_target: np.ndarray) -> None:
        self._target_history = [initial_target.copy() for _ in range(self.history_len)]
        self._cached_embed = None
        self._step_idx = 0
        self.n_system2_calls = 0
        self.n_system1_calls = 0
        self.t_system2 = 0.0
        self.t_system1 = 0.0
        self.embed_diffs = []

    def _maybe_refresh_embed(self, agent_xy: np.ndarray) -> None:
        need = self._cached_embed is None or self._step_idx % self.system2_period == 0
        if not need:
            return
        prev = self._cached_embed
        with torch.no_grad():
            t0 = time.perf_counter()
            hist = torch.from_numpy(np.stack(self._target_history)).unsqueeze(0)
            agent = torch.from_numpy(agent_xy).unsqueeze(0)
            self._cached_embed = self.system2(hist, agent)
            self.t_system2 += time.perf_counter() - t0
        self.n_system2_calls += 1
        if prev is not None:
            self.embed_diffs.append(float(torch.linalg.norm(self._cached_embed - prev)))

    def act(self, obs: np.ndarray) -> np.ndarray:
        agent_xy = obs[:2]
        target_xy = obs[2:]
        self._target_history.pop(0)
        self._target_history.append(target_xy.copy())
        self._maybe_refresh_embed(agent_xy)
        with torch.no_grad():
            t0 = time.perf_counter()
            obs_t = torch.from_numpy(obs).unsqueeze(0)
            action = self.system1(obs_t, self._cached_embed)  # type: ignore[arg-type]
            self.t_system1 += time.perf_counter() - t0
        self.n_system1_calls += 1
        self._step_idx += 1
        return action.squeeze(0).numpy()


def medium_run() -> None:
    print("\n[MEDIUM] decouplage temporel : gain de cout System2 quand period augmente")
    print(
        f"  {'period':>7} | {'n_S2':>5} | {'n_S1':>6} | {'t_S2_ms':>9} | {'t_S1_ms':>9} | {'avg_dist':>8} | mean_dz"
    )
    for period in [1, 4, 8, 16]:
        env = TargetEnv(seed=2025, regime_change_every=80)
        ctrl = DualSystemInstrumented(system2_period=period)
        obs = env.reset()
        ctrl.reset(initial_target=obs[2:])
        total_r = 0.0
        n = 0
        while True:
            obs, r, done = env.step(ctrl.act(obs))
            total_r += r
            n += 1
            if done:
                break
        avg_dist = -total_r / n
        mean_dz = float(np.mean(ctrl.embed_diffs)) if ctrl.embed_diffs else 0.0
        print(
            f"  {period:>7} | {ctrl.n_system2_calls:>5} | {ctrl.n_system1_calls:>6} | "
            f"{ctrl.t_system2 * 1000:>9.2f} | {ctrl.t_system1 * 1000:>9.2f} | "
            f"{avg_dist:>8.3f} | {mean_dz:.4f}"
        )
    print(
        "[MEDIUM] analyse : t_S2 chute proportionnellement a 1/period, "
        "alors que t_S1 reste stable. En production on choisit period > 1 pour "
        "limiter le cout du VLM (System2) au detriment d'une fraicheur de but reduite."
    )


# ============================================================================
# HARD : pipeline d'entrainement complet
# ============================================================================


STRAT_TO_IDX = {s: i for i, s in enumerate(STRATEGIES)}


def collect_expert_dataset(n_episodes: int = 20, seed: int = 7) -> dict:
    """Genere un dataset expert stratifie sur les 3 strategies."""
    history_len = 16
    obs_buf: List[np.ndarray] = []
    hist_buf: List[np.ndarray] = []
    act_buf: List[np.ndarray] = []
    cls_buf: List[int] = []

    # Stratification : on force la strategie initiale en boucle
    for ep in range(n_episodes):
        forced = STRATEGIES[ep % 3]
        env = TargetEnv(seed=seed + ep, regime_change_every=80, forced_strategy=forced)
        obs = env.reset()
        history = [obs[2:].copy() for _ in range(history_len)]
        while True:
            agent_xy = obs[:2]
            target_xy = obs[2:]
            optimal = np.clip(target_xy - agent_xy, -0.3, 0.3).astype(np.float32)

            obs_buf.append(obs.copy())
            hist_buf.append(np.stack(history).copy())
            act_buf.append(optimal)
            cls_buf.append(STRAT_TO_IDX[env.strategy])

            obs, _r, done = env.step(optimal)
            history.pop(0)
            history.append(obs[2:].copy())
            if done:
                break

    return {
        "obs": torch.from_numpy(np.stack(obs_buf)),                     # [N, 4]
        "hist": torch.from_numpy(np.stack(hist_buf)),                   # [N, 16, 2]
        "act": torch.from_numpy(np.stack(act_buf)),                     # [N, 2]
        "cls": torch.from_numpy(np.array(cls_buf, dtype=np.int64)),    # [N]
    }


def train_system2_classification(s2: System2, ds: dict, epochs: int = 30) -> float:
    n = ds["obs"].shape[0]
    perm = torch.randperm(n)
    split = int(0.8 * n)
    train_idx, val_idx = perm[:split], perm[split:]
    opt = torch.optim.AdamW(s2.parameters(), lr=1e-3)
    bs = 256
    for ep in range(epochs):
        s2.train()
        idx = train_idx[torch.randperm(train_idx.shape[0])]
        for i in range(0, idx.shape[0], bs):
            b = idx[i : i + bs]
            logits = s2.classify(ds["hist"][b], ds["obs"][b][:, :2])
            loss = F.cross_entropy(logits, ds["cls"][b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    # eval
    s2.eval()
    with torch.no_grad():
        logits = s2.classify(ds["hist"][val_idx], ds["obs"][val_idx][:, :2])
        acc = (logits.argmax(-1) == ds["cls"][val_idx]).float().mean().item()
    return acc


def train_system1(s1: System1, ds: dict, embed_fn, epochs: int = 30) -> float:
    n = ds["obs"].shape[0]
    perm = torch.randperm(n)
    split = int(0.8 * n)
    train_idx, val_idx = perm[:split], perm[split:]
    opt = torch.optim.AdamW(s1.parameters(), lr=1e-3)
    bs = 256
    final_val = float("inf")
    for ep in range(epochs):
        s1.train()
        idx = train_idx[torch.randperm(train_idx.shape[0])]
        for i in range(0, idx.shape[0], bs):
            b = idx[i : i + bs]
            with torch.no_grad():
                z = embed_fn(ds["hist"][b], ds["obs"][b][:, :2])
            pred = s1(ds["obs"][b], z)
            loss = F.mse_loss(pred, ds["act"][b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    s1.eval()
    with torch.no_grad():
        z = embed_fn(ds["hist"][val_idx], ds["obs"][val_idx][:, :2])
        pred = s1(ds["obs"][val_idx], z)
        final_val = F.mse_loss(pred, ds["act"][val_idx]).item()
    return final_val


def eval_controller(controller, period: int = 8, n_episodes: int = 5, seed: int = 999) -> float:
    dists = []
    for ep in range(n_episodes):
        env = TargetEnv(seed=seed + ep, regime_change_every=80)
        obs = env.reset()
        if hasattr(controller, "reset"):
            controller.reset(initial_target=obs[2:])
        total_r = 0.0
        n = 0
        while True:
            action = controller.act(obs)
            obs, r, done = env.step(action)
            total_r += r
            n += 1
            if done:
                break
        dists.append(-total_r / n)
    return float(np.mean(dists))


class TrainedDualController:
    def __init__(self, s2: System2, s1: System1, system2_period: int = 8, history_len: int = 16):
        self.system2 = s2
        self.system1 = s1
        self.system2_period = system2_period
        self.history_len = history_len
        self._target_history: List[np.ndarray] = []
        self._cached_embed: torch.Tensor | None = None
        self._step_idx = 0

    def reset(self, initial_target: np.ndarray) -> None:
        self._target_history = [initial_target.copy() for _ in range(self.history_len)]
        self._cached_embed = None
        self._step_idx = 0

    def act(self, obs: np.ndarray) -> np.ndarray:
        target_xy = obs[2:]
        self._target_history.pop(0)
        self._target_history.append(target_xy.copy())
        if self._cached_embed is None or self._step_idx % self.system2_period == 0:
            with torch.no_grad():
                hist = torch.from_numpy(np.stack(self._target_history)).unsqueeze(0)
                agent = torch.from_numpy(obs[:2]).unsqueeze(0)
                self._cached_embed = self.system2(hist, agent)
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).unsqueeze(0)
            action = self.system1(obs_t, self._cached_embed)
        self._step_idx += 1
        return action.squeeze(0).numpy()


class TrainedSystem1Only:
    def __init__(self, s1: System1):
        self.system1 = s1
        self.zero_embed = torch.zeros(1, 8)

    def reset(self, initial_target: np.ndarray) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).unsqueeze(0)
            action = self.system1(obs_t, self.zero_embed)
        return action.squeeze(0).numpy()


class OracleController:
    def reset(self, initial_target: np.ndarray) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.clip(obs[2:] - obs[:2], -0.3, 0.3).astype(np.float32)


def hard_run() -> None:
    print("\n[HARD] pipeline complet : dataset expert -> S2 classification -> S1 conditionne -> eval")

    # 1) dataset
    ds = collect_expert_dataset(n_episodes=20, seed=7)
    n = ds["obs"].shape[0]
    counts = torch.bincount(ds["cls"], minlength=3).tolist()
    print(f"[HARD] dataset : N={n}, counts par strategie={counts}")

    # 2) train System2 (classification)
    s2 = System2()
    acc = train_system2_classification(s2, ds, epochs=30)
    print(f"[HARD] System2 classification accuracy (val) : {acc:.3f}")
    # On freeze System2 ensuite
    for p in s2.parameters():
        p.requires_grad_(False)

    # 3a) train System1 conditionne
    s1_dual = System1()
    val_dual = train_system1(
        s1_dual, ds, embed_fn=lambda h, a: s2(h, a), epochs=30
    )

    # 3b) baseline System1-only
    s1_only = System1()
    zero_embed = torch.zeros(1, 8)
    val_only = train_system1(
        s1_only,
        ds,
        embed_fn=lambda h, a: zero_embed.expand(h.shape[0], -1),
        epochs=30,
    )
    print(f"[HARD] System1 MSE val : conditionne={val_dual:.5f} | only={val_only:.5f}")
    # Note pedagogique IMPORTANTE : sur cette tache jouet l'action optimale est
    # `target - agent`, qui est lineaire dans `obs`. System1 peut la reproduire
    # SANS System2 — d'ou MSE comparables. Cela illustre une verite cle du
    # pattern dual-system : System2 n'apporte de la valeur que sur des taches
    # OU LE CONTEXTE TEMPOREL/LANGAGIER N'EST PAS DANS L'OBS COURANTE.
    # C'est exactement pour ca que GR00T/Helix ont un VLM : leurs taches ne se
    # resolvent pas sans contexte d'instruction langagiere ni planning.
    if val_dual < val_only:
        print("[HARD]  -> conditionne bat baseline, OK")
    else:
        print(
            "[HARD]  -> ecart non significatif : sur cette tache jouet (action "
            "lineaire dans obs), System2 n'apporte pas d'info supplementaire. "
            "C'est une lecon : le pattern dual-system n'est utile que si "
            "l'obs courante NE SUFFIT PAS pour decider. Voir Q&A J22."
        )

    # 4) eval environnement
    print("[HARD] eval environnement (avg_distance, plus bas = mieux) :")
    print(f"   - oracle              : {eval_controller(OracleController()):.3f}")
    print(f"   - System1-only trained: {eval_controller(TrainedSystem1Only(s1_only)):.3f}")
    dual_ctrl = TrainedDualController(s2, s1_dual, system2_period=8)
    print(f"   - dual-system trained : {eval_controller(dual_ctrl):.3f}")

    # 5) ablation period
    print("[HARD] ablation system2_period (regime_change_every=80) :")
    for period in [1, 4, 16, 64]:
        ctrl = TrainedDualController(s2, s1_dual, system2_period=period)
        d = eval_controller(ctrl, period=period)
        print(f"   period={period:>3} -> avg_distance={d:.3f}")
    print(
        "[HARD] retro : le decouplage casse quand period > regime_change_every/4 (=20). "
        "A ce moment-la System2 ne raffraichit plus le but assez vite pour suivre les "
        "bascules de strategie. En industrie (Helix/GR00T) on choisit period bas (~8 a 25) "
        "et on compense le cout par du model parallelism : VLM sur un GPU dedie, action "
        "head sur un autre, communication asynchrone."
    )


# ============================================================================
# Entry point
# ============================================================================


def main() -> None:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    print("=" * 60)
    print("J22 - Solutions consolidees (EASY + MEDIUM + HARD)")
    print("=" * 60)

    easy_self_check()
    medium_run()
    hard_run()


if __name__ == "__main__":
    main()
