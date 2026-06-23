"""
Solutions J8 - Motion planning
================================

Trois solutions consolidees :

  - solution_easy()     : collision check + ratio C_free / C empirique vs analytique
  - solution_medium()   : RRT-Connect (deux arbres bidirectionnels)
  - solution_hard()     : RRT* + planning C-space d'un bras 2-DOF planar

# requires: numpy, matplotlib

Reference : LaValle, Planning Algorithms, ch. 5 ; Karaman & Frazzoli 2011 (RRT*).

Lancer :
    python 08-motion-planning.py            # tout
    python 08-motion-planning.py easy       # juste l'easy
    python 08-motion-planning.py medium
    python 08-motion-planning.py hard
"""

from __future__ import annotations

import sys
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


# =============================================================================
# Utilitaires geometriques partages
# =============================================================================

def point_in_rect(p: np.ndarray, rect: tuple[float, float, float, float]) -> bool:
    """Test point dans rectangle axes-aligned (xmin, ymin, xmax, ymax)."""
    # 4 comparaisons elementaires.
    return rect[0] <= p[0] <= rect[2] and rect[1] <= p[1] <= rect[3]


def segment_intersects_rect(a: np.ndarray, b: np.ndarray,
                            rect: tuple, n_steps: int = 20) -> bool:
    """Test segment-rectangle par echantillonnage discret."""
    # On parcourt le segment a pas regulier ; suffisant pour eps petits.
    for t in np.linspace(0.0, 1.0, n_steps):
        p = (1.0 - t) * a + t * b
        if point_in_rect(p, rect):
            return True
    return False


def point_segment_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Distance point-segment 2D : projete p sur [a, b] avec clamp."""
    ab = b - a
    # Norme au carre du segment ; si nulle, segment degenere en point.
    ab2 = float(ab @ ab)
    if ab2 < 1e-12:
        return float(np.linalg.norm(p - a))
    # Parametre de projection clampe sur [0, 1].
    t = float((p - a) @ ab) / ab2
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


# =============================================================================
# SOLUTION EASY : in_collision + ratio empirique vs analytique
# =============================================================================

def solution_easy() -> None:
    print("\n=== EASY : C-space ratio empirique vs analytique ===")

    obstacles = [
        (2.0, 2.0, 4.0, 5.0),
        (6.0, 1.0, 8.0, 4.0),
        (4.5, 6.0, 7.5, 8.0),
    ]

    def in_collision(q, obs_list) -> bool:
        # Si le point est dans un seul obstacle, c'est en collision.
        return any(point_in_rect(np.asarray(q), r) for r in obs_list)

    # 1000 samples uniformes, seed reproductible.
    rng = np.random.default_rng(0)
    samples = rng.uniform(0.0, 10.0, size=(1000, 2))
    free_mask = np.array([not in_collision(s, obstacles) for s in samples])
    n_free = int(free_mask.sum())
    n_obs = 1000 - n_free

    # Ratio analytique : aire totale - aires obstacles.
    aire_obs = sum((r[2] - r[0]) * (r[3] - r[1]) for r in obstacles)
    ratio_analytique = 1.0 - aire_obs / 100.0
    ratio_empirique = n_free / 1000.0

    print(f"  n_free = {n_free} | n_obs = {n_obs}")
    print(f"  ratio empirique  = {ratio_empirique:.3f}")
    print(f"  ratio analytique = {ratio_analytique:.3f}")
    print(f"  ecart            = {abs(ratio_empirique - ratio_analytique):.3f}")

    # Visualisation : libres en bleu, collisions en rouge, obstacles en gris.
    fig, ax = plt.subplots(figsize=(7, 7))
    for r in obstacles:
        ax.add_patch(Rectangle((r[0], r[1]), r[2] - r[0], r[3] - r[1],
                               facecolor="#444", zorder=2))
    ax.scatter(samples[free_mask, 0], samples[free_mask, 1], s=6,
               color="tab:blue", label="C_free")
    ax.scatter(samples[~free_mask, 0], samples[~free_mask, 1], s=8,
               color="tab:red", label="C_obs")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal")
    ax.set_title(f"Easy : ratio C_free empirique = {ratio_empirique:.3f} "
                 f"(analytique {ratio_analytique:.3f})")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()


# =============================================================================
# SOLUTION MEDIUM : RRT-Connect bidirectionnel
# =============================================================================

class TreeRect:
    """Mini-arbre simple pour RRT-Connect, scene a obstacles rectangulaires."""

    def __init__(self, root: np.ndarray, obstacles: list[tuple], eps: float):
        self.nodes = root.reshape(1, 2).copy()
        self.parents: list[int] = [-1]
        self.obstacles = obstacles
        self.eps = eps

    def is_free_segment(self, a: np.ndarray, b: np.ndarray) -> bool:
        return not any(segment_intersects_rect(a, b, r) for r in self.obstacles)

    def nearest_index(self, q: np.ndarray) -> int:
        return int(np.argmin(np.sum((self.nodes - q) ** 2, axis=1)))

    def steer(self, q_near: np.ndarray, q_target: np.ndarray) -> np.ndarray:
        delta = q_target - q_near
        dist = np.linalg.norm(delta)
        if dist < self.eps:
            return q_target.copy()
        return q_near + (delta / dist) * self.eps

    def extend(self, q_target: np.ndarray) -> tuple[str, int]:
        """Avance d'un pas vers q_target. Retourne ('reached'|'advanced'|'trapped', idx)."""
        i_near = self.nearest_index(q_target)
        q_new = self.steer(self.nodes[i_near], q_target)
        if not self.is_free_segment(self.nodes[i_near], q_new):
            return ("trapped", -1)
        self.nodes = np.vstack([self.nodes, q_new])
        self.parents.append(i_near)
        new_idx = self.nodes.shape[0] - 1
        if np.linalg.norm(q_new - q_target) < 1e-6:
            return ("reached", new_idx)
        return ("advanced", new_idx)

    def connect(self, q_target: np.ndarray) -> tuple[str, int]:
        """Repete extend jusqu'a atteindre q_target ou rester bloque."""
        while True:
            status, idx = self.extend(q_target)
            if status != "advanced":
                return status, idx

    def path_to_root(self, idx: int) -> np.ndarray:
        """Remonte les parents depuis idx jusqu'a la racine."""
        out: list[int] = []
        i = idx
        while i != -1:
            out.append(i)
            i = self.parents[i]
        out.reverse()
        return self.nodes[out]


def rrt_connect(q_start, q_goal, bounds, obstacles,
                eps=0.4, max_iter=3000, seed=0):
    """RRT-Connect : deux arbres alternes."""
    rng = np.random.default_rng(seed)
    T_a = TreeRect(np.asarray(q_start, dtype=float), obstacles, eps)
    T_b = TreeRect(np.asarray(q_goal, dtype=float), obstacles, eps)
    swapped = False

    for _ in range(max_iter):
        # Sample uniforme dans la boite.
        q_rand = rng.uniform([bounds[0], bounds[1]], [bounds[2], bounds[3]])
        # Extend T_a vers q_rand.
        status_a, idx_a = T_a.extend(q_rand)
        if status_a != "trapped":
            # Connect T_b vers le nouveau q_new de T_a.
            q_new = T_a.nodes[idx_a]
            status_b, idx_b = T_b.connect(q_new)
            if status_b == "reached":
                # Reconstruire chemin global.
                path_a = T_a.path_to_root(idx_a)
                path_b = T_b.path_to_root(idx_b)
                # T_b est enracine en q_goal : on inverse pour aller jusqu'a q_goal.
                if swapped:
                    full = np.vstack([path_b[::-1], path_a])
                else:
                    full = np.vstack([path_a, path_b[::-1]])
                return full, T_a, T_b, swapped
        # Swap : alterner la pousse pour ne pas qu'un seul arbre fasse tout.
        T_a, T_b = T_b, T_a
        swapped = not swapped

    return None, T_a, T_b, swapped


def solution_medium() -> None:
    print("\n=== MEDIUM : RRT-Connect bidirectionnel ===")

    obstacles = [
        (0.0, 3.5, 7.0, 4.5),
        (3.0, 6.0, 10.0, 7.0),
        (5.0, 8.0, 6.0, 9.5),
    ]
    q_start = np.array([1.0, 1.0])
    q_goal = np.array([9.0, 9.0])
    bounds = (0.0, 0.0, 10.0, 10.0)

    path, T_a, T_b, swapped = rrt_connect(
        q_start, q_goal, bounds, obstacles,
        eps=0.4, max_iter=3000, seed=2026,
    )
    n_total = T_a.nodes.shape[0] + T_b.nodes.shape[0]
    print(f"  n_nodes (T_a + T_b) = {n_total}")
    if path is not None:
        cost = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        print(f"  cost path = {cost:.2f}")
    else:
        print("  echec : pas de chemin trouve.")

    # Visualisation.
    fig, ax = plt.subplots(figsize=(8, 8))
    for r in obstacles:
        ax.add_patch(Rectangle((r[0], r[1]), r[2] - r[0], r[3] - r[1],
                               facecolor="#444", zorder=2))
    # Arbre T_a en bleu, T_b en orange. Note : apres les swaps on perd la trace
    # de "qui est qui". Pour visualiser proprement on retrace la racine de
    # chaque arbre apres execution.
    for tree, color in [(T_a, "tab:blue"), (T_b, "tab:orange")]:
        for i in range(1, tree.nodes.shape[0]):
            p = tree.parents[i]
            if p < 0:
                continue
            a = tree.nodes[p]; b = tree.nodes[i]
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=0.7, alpha=0.7)
    if path is not None:
        ax.plot(path[:, 0], path[:, 1], color="red", linewidth=2.5, zorder=5)
    ax.scatter(*q_start, s=120, color="green", zorder=6, label="start")
    ax.scatter(*q_goal, s=120, color="purple", marker="*", zorder=6, label="goal")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal")
    ax.set_title(f"Medium : RRT-Connect — {n_total} noeuds total")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()


# =============================================================================
# SOLUTION HARD : RRT* + bras 2-DOF planar
# =============================================================================

# --- Bras 2-DOF planar ------------------------------------------------------

L1, L2 = 1.0, 1.0
DISKS = [
    (1.2, 0.8, 0.3),
    (-0.5, 1.5, 0.4),
]


def forward_kinematics(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Retourne (joint_2_position, end_effector_position)."""
    t1, t2 = q[0], q[1]
    j2 = np.array([L1 * math.cos(t1), L1 * math.sin(t1)])
    ee = j2 + np.array([L2 * math.cos(t1 + t2), L2 * math.sin(t1 + t2)])
    return j2, ee


def arm_in_collision(q: np.ndarray) -> bool:
    """Au moins un lien touche un disque ?"""
    j2, ee = forward_kinematics(q)
    base = np.zeros(2)
    for (cx, cy, r) in DISKS:
        c = np.array([cx, cy])
        # Test sur les deux liens (segments base->j2 et j2->ee).
        if point_segment_dist(c, base, j2) < r:
            return True
        if point_segment_dist(c, j2, ee) < r:
            return True
    return False


def arm_segment_in_collision(q_a: np.ndarray, q_b: np.ndarray, n_steps: int = 20) -> bool:
    """Echantillonne le chemin C-space ; un seul q en collision invalide tout."""
    for t in np.linspace(0.0, 1.0, n_steps):
        if arm_in_collision((1.0 - t) * q_a + t * q_b):
            return True
    return False


# --- RRT* generique parametrise par les fonctions de collision -------------

class RRTStar:
    """RRT* generique. Les callbacks is_free_point/is_free_segment laissent
    brancher facilement sur n'importe quel C-space (point 2D, bras, ...)."""

    def __init__(self, q_start, q_goal, bounds, is_free_point, is_free_segment,
                 eps=0.2, goal_bias=0.05, goal_tol=0.2, gamma=1.5,
                 max_iter=2000, seed=0):
        self.q_start = np.asarray(q_start, dtype=float)
        self.q_goal = np.asarray(q_goal, dtype=float)
        self.bounds = bounds
        self.is_free_point = is_free_point
        self.is_free_segment = is_free_segment
        self.eps = eps
        self.goal_bias = goal_bias
        self.goal_tol = goal_tol
        self.gamma = gamma
        self.max_iter = max_iter
        self.rng = np.random.default_rng(seed)

        self.nodes = self.q_start.reshape(1, -1).copy()
        self.parents: list[int] = [-1]
        # Cout = somme des longueurs depuis la racine.
        self.cost: list[float] = [0.0]
        self.goal_idx: int | None = None

    def sample(self):
        if self.rng.random() < self.goal_bias:
            return self.q_goal.copy()
        low = np.array([self.bounds[0], self.bounds[1]])
        high = np.array([self.bounds[2], self.bounds[3]])
        return self.rng.uniform(low, high)

    def nearest_index(self, q):
        return int(np.argmin(np.sum((self.nodes - q) ** 2, axis=1)))

    def steer(self, q_near, q_target):
        delta = q_target - q_near
        dist = float(np.linalg.norm(delta))
        if dist < self.eps:
            return q_target.copy()
        return q_near + (delta / dist) * self.eps

    def neighbor_indices(self, q_new):
        """Voisins dans une boule de rayon dynamique r."""
        n = self.nodes.shape[0]
        # Formule classique en 2D : r ~ gamma * (log(n)/n)^(1/2), borne par 5*eps.
        r = min(self.eps * 5.0, self.gamma * math.sqrt(math.log(max(n, 2)) / max(n, 1)))
        d = np.linalg.norm(self.nodes - q_new, axis=1)
        return np.where(d <= r)[0].tolist(), r

    def plan(self) -> bool:
        for _ in range(self.max_iter):
            q_rand = self.sample()
            i_near = self.nearest_index(q_rand)
            q_near = self.nodes[i_near]
            q_new = self.steer(q_near, q_rand)
            if not self.is_free_point(q_new):
                continue
            if not self.is_free_segment(q_near, q_new):
                continue

            # Choisir le meilleur parent dans le voisinage.
            neigh, _r = self.neighbor_indices(q_new)
            best_parent = i_near
            best_cost = self.cost[i_near] + float(np.linalg.norm(q_new - q_near))
            for j in neigh:
                qj = self.nodes[j]
                c = self.cost[j] + float(np.linalg.norm(q_new - qj))
                if c < best_cost and self.is_free_segment(qj, q_new):
                    best_parent = j
                    best_cost = c

            # Ajouter q_new avec son parent optimal.
            self.nodes = np.vstack([self.nodes, q_new])
            self.parents.append(best_parent)
            self.cost.append(best_cost)
            new_idx = self.nodes.shape[0] - 1

            # Rewiring : pour chaque voisin, voir si passer par q_new ameliore.
            for j in neigh:
                if j == best_parent:
                    continue
                qj = self.nodes[j]
                c_via_new = best_cost + float(np.linalg.norm(qj - q_new))
                if c_via_new < self.cost[j] and self.is_free_segment(q_new, qj):
                    self.parents[j] = new_idx
                    self.cost[j] = c_via_new

            # Critere d'arret : on a vu un noeud assez proche de q_goal et
            # connectable directement. RRT* idealement continue tant que budget
            # restant pour ameliorer le cout — ici on s'arrete des qu'un chemin
            # valide existe pour rester pedagogique.
            if (np.linalg.norm(q_new - self.q_goal) < self.goal_tol
                    and self.is_free_segment(q_new, self.q_goal)):
                self.nodes = np.vstack([self.nodes, self.q_goal])
                self.parents.append(new_idx)
                self.cost.append(best_cost + float(np.linalg.norm(self.q_goal - q_new)))
                self.goal_idx = self.nodes.shape[0] - 1
                return True
        return False

    def extract_path(self):
        if self.goal_idx is None:
            return None
        out: list[int] = []
        i = self.goal_idx
        while i != -1:
            out.append(i)
            i = self.parents[i]
        out.reverse()
        return self.nodes[out]


def solution_hard() -> None:
    print("\n=== HARD : RRT* + bras 2-DOF planar ===")

    q_start = np.array([0.1, 0.1])
    q_goal = np.array([2.5, -1.0])
    # On laisse une boite genereuse en C-space.
    bounds = (-math.pi, -math.pi, 2.0 * math.pi, math.pi)

    rrt = RRTStar(
        q_start, q_goal, bounds,
        is_free_point=lambda q: not arm_in_collision(q),
        is_free_segment=lambda a, b: not arm_segment_in_collision(a, b),
        eps=0.2, goal_bias=0.08, goal_tol=0.2, gamma=2.0,
        max_iter=2000, seed=2026,
    )
    success = rrt.plan()
    print(f"  RRT* success = {success} | n_nodes = {rrt.nodes.shape[0]}")
    path = rrt.extract_path()
    if path is not None:
        cost = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        print(f"  cost C-space = {cost:.3f}")

    # Visualisation 1 : C-space avec arbre + chemin.
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    ax = axes[0]
    for i in range(1, rrt.nodes.shape[0]):
        p = rrt.parents[i]
        if p < 0:
            continue
        a = rrt.nodes[p]; b = rrt.nodes[i]
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#aaa", linewidth=0.6)
    if path is not None:
        ax.plot(path[:, 0], path[:, 1], color="red", linewidth=2.0)
    ax.scatter(*q_start, s=80, color="green", label="start", zorder=5)
    ax.scatter(*q_goal, s=80, color="blue", marker="*", label="goal", zorder=5)
    ax.set_xlabel("theta_1"); ax.set_ylabel("theta_2")
    ax.set_title("Hard : RRT* dans le C-space (theta_1, theta_2)")
    ax.legend(); ax.grid(alpha=0.3); ax.set_aspect("equal")

    # Visualisation 2 : workspace, snapshots du bras le long du chemin.
    ax = axes[1]
    for (cx, cy, r) in DISKS:
        ax.add_patch(Circle((cx, cy), r, facecolor="#888", edgecolor="black"))
    if path is not None and len(path) >= 2:
        # 8 snapshots equireparties pour que la trajectoire soit lisible.
        idxs = np.linspace(0, len(path) - 1, 8, dtype=int)
        cmap = plt.cm.viridis(np.linspace(0, 1, len(idxs)))
        for k, idx in enumerate(idxs):
            j2, ee = forward_kinematics(path[idx])
            ax.plot([0, j2[0], ee[0]], [0, j2[1], ee[1]],
                    color=cmap[k], linewidth=2, marker="o")
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2); ax.set_aspect("equal")
    ax.set_title("Hard : bras 2-DOF dans le workspace, snapshots du chemin")
    ax.grid(alpha=0.3)

    plt.tight_layout(); plt.show()


# =============================================================================
# Dispatcher
# =============================================================================

def main() -> None:
    args = [a.lower() for a in sys.argv[1:]]
    if not args:
        solution_easy(); solution_medium(); solution_hard(); return
    for a in args:
        if a == "easy":
            solution_easy()
        elif a == "medium":
            solution_medium()
        elif a == "hard":
            solution_hard()
        else:
            print(f"Argument inconnu : {a}. Attendu : easy | medium | hard")


if __name__ == "__main__":
    main()
