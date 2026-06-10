"""
Correction commentee — Bras de picking FleetSim : FK/IK + planning articulaire.

Bras planaire RRR en bord de convoyeur : saisir un bac sur la ligne, le deposer
dans la goulotte de tri en passant PAR-DESSUS le muret separateur. Le planner
doit choisir la bonne branche IK et verifier la collision sur TOUTE la
trajectoire — pas seulement aux extremites (le bug de production classique).

Cle de lecture : chaque commentaire explique le POURQUOI. Numpy seul,
deterministe, < 10 s CPU.

Run: python arm_pick_place.py
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Geometrie du poste. Le bras est monte en (0, 0), le convoyeur a droite,
# la goulotte a gauche, le muret separateur entre la base et la goulotte.
# ---------------------------------------------------------------------------
L = np.array([0.40, 0.35, 0.15])     # longueurs des 3 segments (m) — reach 0.90
QDOT_MAX = 1.5                       # vitesse articulaire max (rad/s) — spec moteur
TRAJ_DT = 0.02                       # pas d'echantillonnage trajectoire (50 Hz)
# Muret : rectangle axis-aligned (xmin, xmax, ymin, ymax). Place entre la base
# et la goulotte pour forcer le passage du bras PAR-DESSUS.
OBSTACLE = (-0.24, -0.14, 0.0, 0.34)

PICK = (np.array([0.55, 0.25]), -np.pi / 2)    # bac sur convoyeur, prise verticale
PLACE = (np.array([-0.45, 0.22]), -np.pi / 2)  # goulotte, depose verticale
VIA = (np.array([0.0, 0.62]), np.pi / 2)       # bras leve au-dessus du muret
HOME = (np.array([0.50, 0.40]), -np.pi / 2)    # position de garage


# ---------------------------------------------------------------------------
# 1. Cinematique directe
# ---------------------------------------------------------------------------
def fk(q: np.ndarray) -> tuple[np.ndarray, float]:
    """q (3,) -> (position outil (2,), orientation phi = q1+q2+q3)."""
    pts = joint_positions(q)
    return pts[-1], float(np.sum(q))


def joint_positions(q: np.ndarray) -> np.ndarray:
    """(4, 2) : base, coude, poignet, outil.

    On expose TOUTES les articulations (pas juste l'outil) parce que la
    collision se teste sur les segments du bras : un outil hors du muret
    n'empeche pas le coude de le traverser.
    """
    pts = np.zeros((4, 2))
    angle = 0.0
    for k in range(3):
        angle += q[k]                # angles cumules : chaque axe tourne le reste du bras
        pts[k + 1] = pts[k] + L[k] * np.array([np.cos(angle), np.sin(angle)])
    return pts


def jacobian(q: np.ndarray) -> np.ndarray:
    """Jacobien analytique 3x3 de la tache (x, y, phi).

    Colonne j : effet d'une rotation de l'axe j sur l'outil. Seuls les segments
    EN AVAL de j bougent, d'ou les sommes partielles. La ligne orientation vaut
    1 partout : chaque axe contribue identiquement a phi (bras planaire).
    """
    phis = np.cumsum(q)
    J = np.ones((3, 3))
    for j in range(3):
        J[0, j] = -np.sum(L[j:] * np.sin(phis[j:]))
        J[1, j] = np.sum(L[j:] * np.cos(phis[j:]))
    return J


# ---------------------------------------------------------------------------
# 2. Cinematique inverse — analytique (fermee) et numerique (DLS)
# ---------------------------------------------------------------------------
def _wrap(a: float) -> float:
    """Angle dans [-pi, pi] — indispensable pour comparer des orientations."""
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def ik_analytic(target_xy: np.ndarray, target_phi: float, elbow_up: bool) -> np.ndarray | None:
    """IK fermee : reduction 3R -> 2R via le point poignet.

    L'orientation de l'outil est imposee, donc le poignet est completement
    determine : wrist = target - L3 * direction(phi). Le probleme restant est
    le 2R classique a deux branches (coude haut / coude bas) — c'est le choix
    de branche qui decide de quel cote le coude balaie, donc des collisions.
    """
    wrist = target_xy - L[2] * np.array([np.cos(target_phi), np.sin(target_phi)])
    r2 = float(wrist @ wrist)
    c2 = (r2 - L[0] ** 2 - L[1] ** 2) / (2 * L[0] * L[1])
    if abs(c2) > 1.0 + 1e-12:
        return None                  # poignet hors de l'anneau atteignable du 2R
    c2 = float(np.clip(c2, -1.0, 1.0))
    s2 = np.sqrt(1.0 - c2 ** 2) * (1.0 if elbow_up else -1.0)
    q2 = float(np.arctan2(s2, c2))
    # q1 = direction du poignet, corrigee de l'angle interne du triangle 2R.
    q1 = float(np.arctan2(wrist[1], wrist[0]) - np.arctan2(L[1] * s2, L[0] + L[1] * c2))
    q3 = _wrap(target_phi - q1 - q2)
    return np.array([_wrap(q1), _wrap(q2), q3])


def ik_dls(
    target_xy: np.ndarray,
    target_phi: float,
    q0: np.ndarray,
    max_iters: int = 200,
    tol: float = 1e-6,
) -> np.ndarray | None:
    """IK iterative par damped least squares (Levenberg-Marquardt simplifie).

    Pourquoi l'amortissement : pres d'une singularite (bras tendu), J*J^T est
    quasi-singuliere et la pseudo-inverse pure produit des dq enormes qui font
    diverger l'iteration. Le terme lambda^2*I borne la norme du pas.

    Pourquoi l'amortissement ADAPTATIF (lambda ~ ||err||) : un lambda fixe
    grand rend la convergence finale lineaire et tres lente (des centaines
    d'iterations pres d'une singularite). Loin de la cible on amortit fort
    (robustesse), pres de la cible lambda -> 0 et on retrouve un pas de
    Gauss-Newton quasi-quadratique. C'est le coeur de Levenberg-Marquardt.
    """
    q = q0.astype(float).copy()
    for _ in range(max_iters):
        pos, phi = fk(q)
        err = np.array([target_xy[0] - pos[0], target_xy[1] - pos[1], _wrap(target_phi - phi)])
        err_norm = float(np.linalg.norm(err))
        if err_norm < tol:
            return q
        lam = float(np.clip(0.5 * err_norm, 1e-3, 0.1))
        J = jacobian(q)
        # dq = J^T (J J^T + lambda^2 I)^-1 err — forme "moindres carres amortis".
        dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(3), err)
        q = q + dq
    return None                      # cible hors d'atteinte ou non convergee : pas d'exception


# ---------------------------------------------------------------------------
# 3. Trajectoire quintique + collision
# ---------------------------------------------------------------------------
def quintic_segment(q0: np.ndarray, q1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolation quintique q0 -> q1. Retourne (positions (N,3), vitesses (N,3)).

    Pourquoi quintique et pas lineaire : vitesse ET acceleration nulles aux
    extremites — les moteurs n'encaissent pas de discontinuite de vitesse, et
    les bacs glissent dans la pince si on accelere brutalement.
    La duree T est calee pour que le joint le plus sollicite atteigne PILE
    QDOT_MAX au milieu du segment (vitesse de pointe quintique = 1.875*|dq|/T).
    """
    dq = q1 - q0
    T = max(1.875 * float(np.max(np.abs(dq))) / QDOT_MAX, 0.5)
    n = int(np.ceil(T / TRAJ_DT)) + 1
    tau = np.linspace(0.0, 1.0, n)
    s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
    s_dot = (30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4) / T
    positions = q0[None, :] + s[:, None] * dq[None, :]
    velocities = s_dot[:, None] * dq[None, :]
    return positions, velocities


def arm_hits_obstacle(q: np.ndarray, obstacle: tuple[float, float, float, float]) -> bool:
    """True si un point du bras est dans le rectangle.

    Echantillonnage tous les ~2 cm le long de chaque segment : suffisant ici
    car le muret fait 10 cm d'epaisseur — aucun segment ne peut le traverser
    entre deux echantillons. (Le test exact segment/AABB est l'extension.)
    """
    xmin, xmax, ymin, ymax = obstacle
    pts = joint_positions(q)
    for a, b in zip(pts[:-1], pts[1:]):
        n_samples = max(int(np.linalg.norm(b - a) / 0.02), 2)
        for t in np.linspace(0.0, 1.0, n_samples):
            x, y = a + t * (b - a)
            if xmin <= x <= xmax and ymin <= y <= ymax:
                return True
    return False


def trajectory_collides(traj: np.ndarray, obstacle: tuple[float, float, float, float]) -> bool:
    return any(arm_hits_obstacle(q, obstacle) for q in traj)


def plan_pick_place(
    q_start: np.ndarray, q_pick: np.ndarray, q_place: np.ndarray,
    q_via: np.ndarray, obstacle: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Trajectoire start -> pick -> via -> place. None si une collision subsiste.

    Le via point "bras leve" est la version 0 d'un planner : on encode la
    connaissance metier (le seul passage est au-dessus du muret) plutot que de
    chercher (RRT, cf. J8). C'est exactement ce que fait un integrateur
    FleetSim quand la cellule est simple et certifiable.
    """
    waypoints = [q_start, q_pick, q_via, q_place]
    all_pos, all_vel = [], []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        pos, vel = quintic_segment(a, b)
        if trajectory_collides(pos, obstacle):
            return None
        all_pos.append(pos)
        all_vel.append(vel)
    return np.concatenate(all_pos), np.concatenate(all_vel)


# ---------------------------------------------------------------------------
# Main : verifie chaque critere de reussite par une assertion.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)   # seed fixe : determinisme LogiSim

    # --- Critere 1 : FK de reference (bras tendu a l'horizontale) ---
    pos, phi = fk(np.zeros(3))
    assert np.allclose(pos, [0.90, 0.0]) and phi == 0.0
    print(f"[1] FK([0,0,0]) = ({pos[0]:.2f}, {pos[1]:.2f}), phi={phi:.1f}  OK")

    # --- Critere 2 : roundtrip IK analytique sur 200 cibles atteignables ---
    # On echantillonne des q (donc des cibles garanties atteignables), et on
    # verifie fk(ik(fk(q))) pour les DEUX branches : la branche n'importe pas,
    # seule la tache (x, y, phi) compte.
    worst_pos, worst_phi = 0.0, 0.0
    for _ in range(200):
        q_rand = rng.uniform(-np.pi, np.pi, 3)
        target, target_phi = fk(q_rand)
        for elbow_up in (True, False):
            q_sol = ik_analytic(target, target_phi, elbow_up)
            assert q_sol is not None, "cible atteignable refusee par l'IK"
            p2, phi2 = fk(q_sol)
            worst_pos = max(worst_pos, float(np.linalg.norm(p2 - target)))
            worst_phi = max(worst_phi, abs(_wrap(phi2 - target_phi)))
    assert worst_pos < 1e-9 and worst_phi < 1e-9, (worst_pos, worst_phi)
    print(f"[2] Roundtrip IK analytique x200 x2 branches : pos {worst_pos:.1e} m, "
          f"phi {worst_phi:.1e} rad (< 1e-9)  OK")

    # --- Critere 3 : jacobien analytique vs differences finies ---
    worst_J = 0.0
    eps = 1e-7
    for _ in range(50):
        q_rand = rng.uniform(-np.pi, np.pi, 3)
        J = jacobian(q_rand)
        J_fd = np.zeros((3, 3))
        for j in range(3):
            dq_ = np.zeros(3)
            dq_[j] = eps
            p_plus, phi_plus = fk(q_rand + dq_)
            p_minus, phi_minus = fk(q_rand - dq_)
            J_fd[:2, j] = (p_plus - p_minus) / (2 * eps)
            J_fd[2, j] = (phi_plus - phi_minus) / (2 * eps)
        worst_J = max(worst_J, float(np.max(np.abs(J - J_fd))))
    assert worst_J < 1e-5, worst_J
    print(f"[3] Jacobien analytique vs diff. finies : erreur max {worst_J:.1e} (< 1e-5)  OK")

    # --- Critere 4 : DLS converge sur cibles atteignables, None hors d'atteinte ---
    for _ in range(50):
        q_rand = rng.uniform(-np.pi, np.pi, 3)
        target, target_phi = fk(q_rand)
        q_sol = ik_dls(target, target_phi, q0=np.array([0.3, 0.3, 0.3]))
        assert q_sol is not None, f"DLS n'a pas converge vers {target}"
    assert ik_dls(np.array([1.2, 0.0]), 0.0, q0=np.array([0.3, 0.3, 0.3])) is None
    print("[4] DLS : 50/50 cibles atteignables convergees (< 200 iters), "
          "None a 1.2 m (reach 0.90)  OK")

    # --- Configurations de la mission pick & place ---
    q_home = ik_analytic(*HOME, elbow_up=True)
    q_pick = ik_analytic(*PICK, elbow_up=True)
    q_via = ik_analytic(*VIA, elbow_up=True)
    q_place = ik_analytic(*PLACE, elbow_up=True)
    assert all(q is not None for q in (q_home, q_pick, q_via, q_place))

    # --- Critere 6 (le bug de production) : la branche coude BAS balaie le muret ---
    # Les deux branches IK atteignent exactement la meme cible outil, mais le
    # coude ne passe pas du meme cote : coude bas, il plonge dans le muret.
    # C'est pour ca qu'un planner ne peut pas se contenter de "l'IK a converge".
    q_place_down = ik_analytic(*PLACE, elbow_up=False)
    assert q_place_down is not None
    assert arm_hits_obstacle(q_place_down, OBSTACLE), (
        "la config coude bas devrait toucher le muret — la geometrie du poste a change ?"
    )
    direct_down, _ = quintic_segment(q_pick, q_place_down)
    assert trajectory_collides(direct_down, OBSTACLE)
    print("[6] Branche coude bas a la depose : collision muret (config ET trajectoire) — "
          "le bug que le choix de branche corrige  OK")

    # --- Critere 5 : la trajectoire planifiee est propre ---
    planned = plan_pick_place(q_home, q_pick, q_place, q_via, OBSTACLE)
    assert planned is not None, "le planner n'a pas trouve de trajectoire sans collision"
    traj, vels = planned

    # Precision aux waypoints : l'outil passe exactement par pick / via / place.
    for q_wp, (target, target_phi) in zip((q_pick, q_via, q_place), (PICK, VIA, PLACE)):
        p_wp, phi_wp = fk(q_wp)
        assert np.linalg.norm(p_wp - target) < 1e-6 and abs(_wrap(phi_wp - target_phi)) < 1e-6

    # Vitesse : bornee par la spec moteur, nulle aux extremites (quintique).
    vmax = float(np.max(np.abs(vels)))
    assert vmax <= QDOT_MAX * 1.001, vmax
    assert float(np.max(np.abs(vels[0]))) < 1e-9 and float(np.max(np.abs(vels[-1]))) < 1e-9

    # Zero collision sur toute la trajectoire echantillonnee.
    assert not trajectory_collides(traj, OBSTACLE)
    print(f"[5] Trajectoire planifiee : {len(traj)} pas, waypoints atteints (< 1e-6), "
          f"vitesse max {vmax:.3f} rad/s (<= {QDOT_MAX}), zero collision  OK")

    print("\nTous les criteres de reussite passent.")
