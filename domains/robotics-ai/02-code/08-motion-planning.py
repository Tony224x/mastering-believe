"""
J8 - Motion planning : RRT 2D from scratch
============================================

Objectif pedagogique :
    - Implementer RRT en 2D avec obstacles rectangulaires
    - Visualiser l'arborescence rapidement-exploree (Voronoi-bias)
    - Tracer le chemin solution une fois q_goal atteint

Reference : LaValle, Planning Algorithms, ch. 5.5 (RRT).

# requires: numpy, matplotlib

Lancer :
    python 08-motion-planning.py
"""

from __future__ import annotations

# numpy : sampling, distances, operations vectorielles sur les noeuds.
import numpy as np

# matplotlib : visualiser l'arbre RRT en pleine croissance et le chemin final.
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# -----------------------------------------------------------------------------
# Environnement : obstacles rectangulaires alignes sur les axes
# -----------------------------------------------------------------------------

class RectObstacle:
    """Rectangle aligne axes-aligned : (x_min, y_min, x_max, y_max)."""

    def __init__(self, x_min: float, y_min: float, x_max: float, y_max: float):
        # On stocke les coins pour des tests d'inclusion peu couteux.
        self.x_min, self.y_min = x_min, y_min
        self.x_max, self.y_max = x_max, y_max

    def contains(self, p: np.ndarray) -> bool:
        """Le point p est-il a l'interieur du rectangle ?"""
        # Comparaisons componentwise : appartenance dans le rectangle ferme.
        return (self.x_min <= p[0] <= self.x_max
                and self.y_min <= p[1] <= self.y_max)

    def segment_intersects(self, a: np.ndarray, b: np.ndarray, n_steps: int = 20) -> bool:
        """Test de collision segment-rectangle par echantillonnage discret.

        Pourquoi un test discret plutot qu'analytique : c'est generique, marche
        pour des obstacles arbitraires (capsules, mesh) et reste correct pour
        des pas RRT typiques (eps petit) — n_steps=20 suffit largement.
        """
        # Interpolation t in [0, 1], on teste chaque point intermediaire.
        for t in np.linspace(0.0, 1.0, n_steps):
            # Le point sur le segment a la fraction t.
            p = (1.0 - t) * a + t * b
            if self.contains(p):
                # On a trouve un point en collision : segment invalide.
                return True
        # Aucun point en collision.
        return False


# -----------------------------------------------------------------------------
# RRT : structure de donnees et algorithme
# -----------------------------------------------------------------------------

class RRT:
    """Rapidly-exploring Random Tree en 2D.

    Convention : les noeuds sont stockes dans un tableau (n, 2) numpy ;
    les parents dans un tableau d'indices (le noeud i a pour parent parent[i]).
    parent[0] = -1 par convention pour la racine.
    """

    def __init__(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        bounds: tuple[float, float, float, float],  # (x_min, y_min, x_max, y_max)
        obstacles: list[RectObstacle],
        eps: float = 0.5,        # pas d'avancee (steer)
        goal_bias: float = 0.05,  # proba de tirer q_goal directement
        goal_tol: float = 0.5,    # rayon de connexion au but
        max_iter: int = 5000,
        seed: int | None = 42,
    ):
        # Limites du C-space (ici workspace = C-space pour un robot ponctuel 2D).
        self.bounds = bounds
        # On garde les obstacles pour les tests de collision.
        self.obstacles = obstacles
        # Hyperparametres RRT classiques.
        self.eps = eps
        self.goal_bias = goal_bias
        self.goal_tol = goal_tol
        self.max_iter = max_iter
        # Configurations de depart et d'arrivee.
        self.q_start = np.asarray(q_start, dtype=float)
        self.q_goal = np.asarray(q_goal, dtype=float)
        # RNG reproductible pour pouvoir analyser le meme run plusieurs fois.
        self.rng = np.random.default_rng(seed)

        # Stockage des noeuds : on commence avec juste la racine q_start.
        # Format (n, 2) pour profiter des distances vectorisees numpy.
        self.nodes = self.q_start.reshape(1, 2).copy()
        # parent[i] = indice du parent de nodes[i] ; -1 pour la racine.
        self.parents: list[int] = [-1]
        # Indice du noeud final si la solution est trouvee.
        self.goal_idx: int | None = None

    # -------------------------------------------------------------------------
    # Primitives de base
    # -------------------------------------------------------------------------

    def is_free_point(self, q: np.ndarray) -> bool:
        """Le point q est-il dans C_free (pas dans un obstacle) ?"""
        # any() court-circuite des qu'un obstacle contient le point.
        return not any(obs.contains(q) for obs in self.obstacles)

    def is_free_segment(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Le segment [a, b] reste-t-il dans C_free ?"""
        # Si un seul obstacle intersecte le segment, c'est invalide.
        return not any(obs.segment_intersects(a, b) for obs in self.obstacles)

    def sample(self) -> np.ndarray:
        """Echantillon uniforme dans le C-space, avec biais vers le but.

        goal_bias : trick standard de LaValle pour accelerer la convergence.
        Sans ce biais, RRT converge mais lentement vers q_goal precisement.
        """
        # Avec proba goal_bias on tire q_goal pour pousser l'arbre vers le but.
        if self.rng.random() < self.goal_bias:
            return self.q_goal.copy()
        # Sinon, sample uniforme dans la boite englobante.
        x = self.rng.uniform(self.bounds[0], self.bounds[2])
        y = self.rng.uniform(self.bounds[1], self.bounds[3])
        return np.array([x, y])

    def nearest_index(self, q: np.ndarray) -> int:
        """Indice du noeud existant le plus proche de q.

        Pour rester pedagogique : O(n) brute force. En production on utiliserait
        un kd-tree (scipy.spatial.cKDTree) pour passer a O(log n).
        """
        # Distances au carre (sqrt inutile pour comparer).
        d2 = np.sum((self.nodes - q) ** 2, axis=1)
        # argmin renvoie l'indice du minimum.
        return int(np.argmin(d2))

    def steer(self, q_near: np.ndarray, q_target: np.ndarray) -> np.ndarray:
        """Avance d'un pas eps depuis q_near en direction de q_target.

        Si q_target est plus proche que eps, on prend q_target tel quel —
        utile pour la connection finale au but.
        """
        # Vecteur direction et sa norme.
        delta = q_target - q_near
        dist = np.linalg.norm(delta)
        # Si on est deja a portee, on prend le point cible.
        if dist < self.eps:
            return q_target.copy()
        # Sinon on avance d'exactement eps dans la bonne direction.
        return q_near + (delta / dist) * self.eps

    # -------------------------------------------------------------------------
    # Boucle principale RRT
    # -------------------------------------------------------------------------

    def plan(self) -> bool:
        """Lance la boucle RRT. Retourne True si q_goal a ete atteint."""
        for _ in range(self.max_iter):
            # 1. Sample (avec goal-bias).
            q_rand = self.sample()
            # 2. Trouver le noeud le plus proche dans l'arbre.
            i_near = self.nearest_index(q_rand)
            q_near = self.nodes[i_near]
            # 3. Steer : avance d'un pas eps vers q_rand.
            q_new = self.steer(q_near, q_rand)
            # 4. Test de collision sur le segment q_near -> q_new.
            if not self.is_free_segment(q_near, q_new):
                # Edge en collision : on jette ce sample, on continue.
                continue
            # 5. Edge valide : on ajoute q_new dans l'arbre.
            self.nodes = np.vstack([self.nodes, q_new])
            self.parents.append(i_near)
            new_idx = self.nodes.shape[0] - 1
            # 6. Critere d'arret : on est assez proche de q_goal ET le segment
            #    de connexion final est libre.
            if (np.linalg.norm(q_new - self.q_goal) < self.goal_tol
                    and self.is_free_segment(q_new, self.q_goal)):
                # On ajoute explicitement q_goal comme dernier noeud, parent = q_new.
                self.nodes = np.vstack([self.nodes, self.q_goal])
                self.parents.append(new_idx)
                self.goal_idx = self.nodes.shape[0] - 1
                return True
        # Pas trouve dans le budget alloue : echec.
        return False

    def extract_path(self) -> np.ndarray | None:
        """Remonte l'arbre depuis q_goal jusqu'a q_start.

        Retourne un tableau (k, 2) ou None si pas de solution.
        """
        if self.goal_idx is None:
            return None
        # On remonte parent par parent.
        path_idx: list[int] = []
        i = self.goal_idx
        while i != -1:
            path_idx.append(i)
            i = self.parents[i]
        # On inverse pour avoir start -> goal.
        path_idx.reverse()
        return self.nodes[path_idx]


# -----------------------------------------------------------------------------
# Visualisation : arbre + chemin + obstacles
# -----------------------------------------------------------------------------

def visualize(rrt: RRT, path: np.ndarray | None, title: str = "RRT 2D") -> None:
    """Trace l'arbre RRT, le chemin solution, les obstacles, start et goal."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Cadre du C-space pour delimiter visuellement la zone de planning.
    x_min, y_min, x_max, y_max = rrt.bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")

    # Obstacles en gris fonce — la zone interdite C_obs.
    for obs in rrt.obstacles:
        rect = Rectangle(
            (obs.x_min, obs.y_min),
            obs.x_max - obs.x_min,
            obs.y_max - obs.y_min,
            facecolor="#444444",
            edgecolor="black",
            zorder=2,
        )
        ax.add_patch(rect)

    # Aretes de l'arbre : on les trace toutes en gris clair pour montrer
    # comment RRT colonise C_free (bias-Voronoi visible).
    for i in range(1, rrt.nodes.shape[0]):
        parent_idx = rrt.parents[i]
        if parent_idx < 0:
            continue
        a = rrt.nodes[parent_idx]
        b = rrt.nodes[i]
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#cccccc", linewidth=0.8, zorder=1)

    # Noeuds de l'arbre — petits points pour la lisibilite.
    ax.scatter(rrt.nodes[:, 0], rrt.nodes[:, 1], s=4, color="#888888", zorder=3)

    # Chemin solution en rouge epais — superpose sur l'arbre.
    if path is not None:
        ax.plot(path[:, 0], path[:, 1], color="red", linewidth=2.5, zorder=5,
                label=f"path ({len(path)} noeuds)")

    # Marqueurs start (vert) et goal (bleu) — gros pour qu'on les voie de loin.
    ax.scatter(*rrt.q_start, s=120, color="green", marker="o", zorder=6, label="start")
    ax.scatter(*rrt.q_goal, s=120, color="blue", marker="*", zorder=6, label="goal")

    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Demo : labyrinthe 10x10 avec quelques obstacles
# -----------------------------------------------------------------------------

def main() -> None:
    # Une scene simple mais non-triviale : un couloir a slalomer.
    obstacles = [
        # Mur median bas avec une ouverture a droite.
        RectObstacle(0.0, 3.5, 7.0, 4.5),
        # Mur median haut avec une ouverture a gauche.
        RectObstacle(3.0, 6.0, 10.0, 7.0),
        # Petit obstacle isole au centre-haut pour forcer un detour.
        RectObstacle(5.0, 8.0, 6.0, 9.5),
    ]

    rrt = RRT(
        q_start=np.array([1.0, 1.0]),     # coin bas-gauche
        q_goal=np.array([9.0, 9.0]),       # coin haut-droit
        bounds=(0.0, 0.0, 10.0, 10.0),     # C-space = [0, 10]^2
        obstacles=obstacles,
        eps=0.4,
        goal_bias=0.08,
        goal_tol=0.4,
        max_iter=5000,
        seed=2026,
    )

    # On lance et on imprime quelques metriques utiles pour comprendre le run.
    success = rrt.plan()
    n_nodes = rrt.nodes.shape[0]
    print(f"[RRT] success={success} | n_nodes={n_nodes}")

    if success:
        path = rrt.extract_path()
        # Cout = somme des longueurs euclidiennes des aretes du chemin.
        dists = np.linalg.norm(np.diff(path, axis=0), axis=1)
        print(f"[RRT] path length = {dists.sum():.2f} | n_waypoints = {len(path)}")
        visualize(rrt, path, title=f"RRT 2D — {n_nodes} noeuds, chemin {dists.sum():.2f}")
    else:
        print("[RRT] echec : augmenter max_iter ou ajuster eps / goal_bias.")
        visualize(rrt, None, title="RRT 2D — echec")


if __name__ == "__main__":
    main()
