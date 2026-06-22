"""
J22 — Frontier humanoid : mini dual-system jouet (System1 + System2).

Objectif pedagogique
--------------------
Reproduire en miniature le pattern industry 2025-2026 (GR00T N1, Helix, LBM TRI) :
- System2 : reseau LENT qui regarde la scene haut-niveau et produit un GOAL EMBEDDING
            (analogue d'un VLM 7B qui interprete une instruction et planifie).
- System1 : reseau RAPIDE qui prend (etat courant + goal embedding) et sort les actions
            de controle continu (analogue d'une action head a 200 Hz).

Tache jouet
-----------
Un agent point en 2D doit poursuivre une cible mobile.
- L'environnement change la "strategie" cible toutes les K steps : parfois la cible
  oscille horizontalement, parfois elle decrit un cercle, parfois elle est statique.
- System2 observe l'historique de la cible et infere la strategie courante (3 modes),
  produit un goal embedding qui resume "quoi viser dans les prochaines steps".
- System1 prend (position agent, position cible, goal embedding) et sort dx, dy.

Pourquoi ca illustre le pattern
-------------------------------
- System2 tourne a 1/8 de la frequence de System1 (decoupage temporel).
- System1 ne voit jamais l'historique brut : il consomme un latent deja distille.
- On peut swapper System2 sans toucher System1 (embodiment-agnostic dans l'esprit).

Verification : `python -m py_compile 22-frontier-humanoid.py` doit PASS.
Run : `python 22-frontier-humanoid.py` lance une demo console (pas de dependance graphique).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1) Environnement jouet : cible 2D avec strategie qui change
# ---------------------------------------------------------------------------

STRATEGIES = ("static", "horizontal_oscillation", "circle")


@dataclass
class TargetEnv:
    """Cible 2D mobile dont la strategie change toutes les `regime_change_every` steps.

    L'agent (point) doit minimiser ||agent - target||. C'est un proxy pour
    'humanoide qui doit suivre un sous-but qui change'.
    """

    regime_change_every: int = 80   # tous les 80 pas, on change de strategie
    box_size: float = 5.0
    seed: int = 42
    # etat
    t: int = 0
    strategy: str = "static"
    target_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    agent_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    rng: random.Random = field(default_factory=random.Random)

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
        # On choisit une nouvelle strategie. Cle : System2 doit deviner laquelle
        # depuis l'historique recent (sans qu'on lui donne l'etiquette directement).
        self.strategy = self.rng.choice(STRATEGIES)
        # parametres de la strategie courante
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
        # On clamp l'action pour rester dans des bornes physiques (analogue couples max)
        action = np.clip(action, -0.3, 0.3).astype(np.float32)
        self.agent_xy = self.agent_xy + action
        # Borne la box
        self.agent_xy = np.clip(self.agent_xy, -self.box_size, self.box_size)

        self.t += 1
        if self.t % self.regime_change_every == 0:
            self._pick_new_strategy()
        self._update_target()

        dist = float(np.linalg.norm(self.agent_xy - self.target_xy))
        reward = -dist  # minimiser la distance
        done = self.t >= 1000
        return self._obs(), reward, done

    def _obs(self) -> np.ndarray:
        # Observation atomique : ce que System1 voit a chaque step
        # (positions, pas d'historique — l'historique est gere par System2 plus haut)
        return np.concatenate([self.agent_xy, self.target_xy]).astype(np.float32)


# ---------------------------------------------------------------------------
# 2) System2 : reseau LENT qui digere l'historique et sort un goal embedding
# ---------------------------------------------------------------------------


class System2(nn.Module):
    """Mock VLM/planner : MLP profond sur historique de positions cible.

    Dans la vraie vie c'est un VLM 7B (Eagle pour GR00T, ~7B pour Helix). Ici on
    simule le cout par un sleep/delai conceptuel : System2 ne tourne qu'a 1/K.
    """

    def __init__(self, history_len: int = 16, hidden: int = 64, embed_dim: int = 8):
        super().__init__()
        self.history_len = history_len
        self.embed_dim = embed_dim
        # input = history_len * 2 (xy de la cible) + 2 (xy agent courant)
        self.net = nn.Sequential(
            nn.Linear(history_len * 2 + 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim),
            nn.Tanh(),  # latent borne, comme un embedding normalise
        )

    def forward(self, target_history: torch.Tensor, agent_xy: torch.Tensor) -> torch.Tensor:
        # target_history : [B, history_len, 2]
        # agent_xy       : [B, 2]
        b = target_history.shape[0]
        x = torch.cat([target_history.reshape(b, -1), agent_xy], dim=-1)
        return self.net(x)


# ---------------------------------------------------------------------------
# 3) System1 : reseau RAPIDE qui consomme (obs, goal_embed) et sort des actions
# ---------------------------------------------------------------------------


class System1(nn.Module):
    """Action head rapide : MLP leger conditionne par le goal embedding System2.

    Equivalent fonctionnel du DiT de GR00T ou du MLP 80M de Helix : il ne comprend
    pas le langage, il consomme une representation deja digeree.
    """

    def __init__(self, obs_dim: int = 4, embed_dim: int = 8, hidden: int = 32, action_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh(),  # actions bornees [-1, 1] avant scaling externe
        )
        self.action_scale = 0.3

    def forward(self, obs: torch.Tensor, goal_embed: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, goal_embed], dim=-1)
        return self.net(x) * self.action_scale


# ---------------------------------------------------------------------------
# 4) Dual-system controller : orchestre System2 (lent) et System1 (rapide)
# ---------------------------------------------------------------------------


class DualSystemController:
    """Pattern industry 2025 : System2 a basse frequence, System1 a haute frequence.

    En production :
    - System2 tourne sur GPU lent (~7 Hz pour Helix / GR00T)
    - System1 tourne sur DSP/edge ou GPU dedie (~200 Hz)
    On reproduit ca en n'appelant System2 que tous les `system2_period` steps.
    """

    def __init__(self, history_len: int = 16, system2_period: int = 8):
        self.system2 = System2(history_len=history_len)
        self.system1 = System1()
        self.history_len = history_len
        self.system2_period = system2_period
        self._target_history: list[np.ndarray] = []
        self._cached_embed: torch.Tensor | None = None
        self._step_idx = 0

    def reset(self, initial_target: np.ndarray) -> None:
        # Padding d'historique avec la position initiale (warm-up)
        self._target_history = [initial_target.copy() for _ in range(self.history_len)]
        self._cached_embed = None
        self._step_idx = 0

    def _maybe_refresh_embed(self, agent_xy: np.ndarray) -> None:
        """Recalcule le goal embedding seulement quand System2 doit tourner."""
        need = self._cached_embed is None or self._step_idx % self.system2_period == 0
        if not need:
            return
        with torch.no_grad():
            hist = torch.from_numpy(np.stack(self._target_history)).unsqueeze(0)  # [1, H, 2]
            agent = torch.from_numpy(agent_xy).unsqueeze(0)  # [1, 2]
            self._cached_embed = self.system2(hist, agent)  # [1, embed_dim]

    def act(self, obs: np.ndarray) -> np.ndarray:
        """obs = [agent_x, agent_y, target_x, target_y]"""
        agent_xy = obs[:2]
        target_xy = obs[2:]

        # Mise a jour de l'historique des positions cibles
        self._target_history.pop(0)
        self._target_history.append(target_xy.copy())

        # System2 ne tourne qu'a basse frequence
        self._maybe_refresh_embed(agent_xy)

        # System1 tourne a chaque step
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).unsqueeze(0)
            action = self.system1(obs_t, self._cached_embed)  # type: ignore[arg-type]
        self._step_idx += 1
        return action.squeeze(0).numpy()


# ---------------------------------------------------------------------------
# 5) Baseline naive : System1 SEUL (pas de System2), pour comparer
# ---------------------------------------------------------------------------


class System1Only:
    """Baseline : on appelle System1 mais en lui filant un goal_embed nul.

    Sert a illustrer en quoi System2 apporte de la valeur (au moins en theorie).
    Note pedagogique : ces reseaux ne sont PAS entraines ici — l'objectif est
    d'exposer le pattern architectural et les flux de donnees, pas de battre un
    benchmark. Voir J24-J28 pour le training reel sur PushT.
    """

    def __init__(self):
        self.system1 = System1()
        self.zero_embed = torch.zeros(1, 8)

    def reset(self, initial_target: np.ndarray) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).unsqueeze(0)
            action = self.system1(obs_t, self.zero_embed)
        return action.squeeze(0).numpy()


# ---------------------------------------------------------------------------
# 6) Demo console
# ---------------------------------------------------------------------------


def run_episode(controller, env: TargetEnv, label: str) -> float:
    obs = env.reset()
    if hasattr(controller, "reset"):
        controller.reset(initial_target=obs[2:])
    total_r = 0.0
    n_steps = 0
    s2_calls = 0
    while True:
        # Petit hack pour compter les calls System2 si dispo
        before = getattr(controller, "_step_idx", None)
        action = controller.act(obs)
        if before is not None and (controller._step_idx - 1) % controller.system2_period == 0:
            s2_calls += 1

        obs, reward, done = env.step(action)
        total_r += reward
        n_steps += 1
        if done:
            break

    avg_dist = -total_r / n_steps
    extra = f", System2 calls={s2_calls}" if s2_calls else ""
    print(f"[{label}] steps={n_steps}, avg_distance={avg_dist:.3f}{extra}")
    return avg_dist


def main() -> None:
    # Reproductibilite
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    print("=" * 60)
    print("J22 - Dual-system jouet : System2 (lent) + System1 (rapide)")
    print("=" * 60)

    env = TargetEnv(seed=123, regime_change_every=80)

    # 1) Baseline : System1 seul (pas d'historique digere)
    baseline = System1Only()
    run_episode(baseline, env, "baseline System1-only (no System2)")

    # 2) Dual-system : System2 toutes les 8 steps, System1 chaque step
    dual = DualSystemController(history_len=16, system2_period=8)
    run_episode(dual, env, "dual-system  System2@1/8 + System1@1/1")

    # 3) Dual-system mais System2 a chaque step (cher, demonstratif)
    dual_eager = DualSystemController(history_len=16, system2_period=1)
    run_episode(dual_eager, env, "dual-system  System2@1/1 (expensive)")

    print()
    print("Note importante : ces reseaux ne sont PAS entraines.")
    print("Sans entrainement, les reseaux sortent du bruit et la distance")
    print("moyenne est elevee (~ box_size). L'objectif du module est d'exposer")
    print("le PATTERN architectural (decouplage temporel, flux de donnees,")
    print("interface latente entre les deux systemes), pas la performance.")
    print()
    print("Pour aller plus loin :")
    print("  - J24-J28 capstone : entrainement reel d'une diffusion policy (==System1)")
    print("  - Brancher un VLM (Llama 3 / Qwen-VL) en amont pour faire un vrai System2")
    print("  - Voir REFERENCES.md #15 (GR00T N1) et #16 (Helix) pour l'industrie")


if __name__ == "__main__":
    main()
