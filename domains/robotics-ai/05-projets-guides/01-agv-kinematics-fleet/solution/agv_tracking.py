"""
Correction commentee — AGV differentiel FleetSim : cinematique + pure pursuit.

Objectif : transformer un chemin (sorti du pathfinder) en commandes de roues
gauche/droite, avec une erreur de suivi bornee et certifiable.

Cle de lecture : chaque commentaire explique le POURQUOI (choix de design),
pas le QUOI (lisible dans le code). Numpy seul, deterministe, < 10 s CPU.

Run: python agv_tracking.py
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Parametres physiques et de mission. Valeurs typiques d'un AGV de transport
# de palettes (1 m/s nominal, empattement 0.5 m). V_WHEEL_MAX > v_nominal pour
# laisser de la marge a la roue exterieure dans les virages.
# ---------------------------------------------------------------------------
WHEEL_BASE = 0.5      # distance entre les deux roues motrices (m)
V_WHEEL_MAX = 1.5     # vitesse max d'une roue (m/s) — limite moteur
V_NOMINAL = 1.0       # vitesse de croisiere en ligne droite (m/s)
LOOKAHEAD = 0.6       # distance d'anticipation pure pursuit (m)
DT = 0.02             # tick de simulation 50 Hz (cf. moteur FleetSim 10-50 Hz)
PATH_STEP = 0.05      # pas de reechantillonnage du chemin (m)


# ---------------------------------------------------------------------------
# 1. Cinematique differentielle
# ---------------------------------------------------------------------------
def wheels_to_body(v_left: float, v_right: float, wheel_base: float) -> tuple[float, float]:
    """(vitesses roues) -> (v, omega) du chassis.

    Le chassis tourne autour de l'ICC : la difference des vitesses de roues
    cree la rotation, leur moyenne cree la translation. C'est LE modele de
    base de toute base differentielle (cf. theorie J3).
    """
    v = 0.5 * (v_right + v_left)
    omega = (v_right - v_left) / wheel_base
    return v, omega


def body_to_wheels(v: float, omega: float, wheel_base: float) -> tuple[float, float]:
    """Inverse exact de wheels_to_body — le controleur raisonne en (v, omega),
    les moteurs ne comprennent que des vitesses de roues."""
    v_left = v - 0.5 * omega * wheel_base
    v_right = v + 0.5 * omega * wheel_base
    return v_left, v_right


def saturate_wheels(v: float, omega: float) -> tuple[float, float]:
    """Sature les roues a V_WHEEL_MAX en PRESERVANT la courbure.

    Pourquoi pas un simple clip par roue : clipper une seule roue change le
    ratio omega/v, donc la courbure -> l'AGV sort de son couloir au pire
    moment (en virage). On scale les DEUX roues du meme facteur : l'AGV
    ralentit mais reste sur son arc.
    """
    v_l, v_r = body_to_wheels(v, omega, WHEEL_BASE)
    peak = max(abs(v_l), abs(v_r))
    if peak > V_WHEEL_MAX:
        scale = V_WHEEL_MAX / peak
        v_l *= scale
        v_r *= scale
    return wheels_to_body(v_l, v_r, WHEEL_BASE)


def integrate_pose(pose: np.ndarray, v: float, omega: float, dt: float) -> np.ndarray:
    """Pose (x, y, theta) apres dt — integration EXACTE par arc de cercle.

    Pourquoi pas Euler : a (v, omega) constants l'AGV decrit un arc parfait
    de rayon R = v/omega autour de l'ICC. Euler remplace l'arc par sa corde
    tangente -> derive systematique vers l'exterieur, cumulable sur un shift
    de 8 h. La forme fermee est exacte ET pas plus chere.
    """
    x, y, theta = pose
    if abs(omega) < 1e-9:
        # Limite ligne droite : R -> inf, la formule en arc divise par zero.
        return np.array([x + v * dt * np.cos(theta), y + v * dt * np.sin(theta), theta])
    radius = v / omega
    theta_new = theta + omega * dt
    x_new = x + radius * (np.sin(theta_new) - np.sin(theta))
    y_new = y - radius * (np.cos(theta_new) - np.cos(theta))
    return np.array([x_new, y_new, theta_new])


def integrate_pose_euler(pose: np.ndarray, v: float, omega: float, dt: float) -> np.ndarray:
    """Euler explicite — garde uniquement pour DEMONTRER la derive (etape 2)."""
    x, y, theta = pose
    return np.array([x + v * dt * np.cos(theta), y + v * dt * np.sin(theta), theta + omega * dt])


# ---------------------------------------------------------------------------
# 2. Chemin d'entrepot : boucle rectangulaire reechantillonnee dense
# ---------------------------------------------------------------------------
def resample_path(waypoints: np.ndarray, step: float) -> np.ndarray:
    """Interpole les waypoints en points espaces de `step`.

    Pourquoi : pure pursuit cherche "le point a distance lookahead" — sur des
    waypoints espaces de plusieurs metres ce point n'existe pas, sur un chemin
    dense (5 cm) une recherche du plus proche suffit.
    """
    points = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        seg_len = float(np.linalg.norm(b - a))
        n = max(int(seg_len / step), 1)
        for i in range(n):
            points.append(a + (b - a) * (i / n))
    return np.array(points)


def warehouse_loop() -> np.ndarray:
    """Boucle d'allees 20 m x 10 m, coins droits — pire cas pour le suivi."""
    waypoints = np.array([
        [0.0, 0.0], [20.0, 0.0], [20.0, 10.0], [0.0, 10.0], [0.0, 0.0],
    ])
    return resample_path(waypoints, PATH_STEP)


# ---------------------------------------------------------------------------
# 3. Pure pursuit
# ---------------------------------------------------------------------------
def pure_pursuit_step(
    pose: np.ndarray, path: np.ndarray, progress_idx: int, lookahead: float, v_nominal: float
) -> tuple[float, float, int]:
    """Retourne (v, omega, nouveau progress_idx).

    progress_idx = index du dernier point "atteint" : on ne cherche le plus
    proche que dans une fenetre DEVANT lui. Pourquoi : sur une boucle, le
    point globalement le plus proche peut etre sur l'allee d'en face (10 m a
    vol d'oiseau) — l'AGV ferait demi-tour. La progression monotone encode
    "on suit le plan du pathfinder", pas "on va au plus pres".
    """
    n = len(path)
    window = 200  # ~10 m de chemin a 5 cm/pt : large devant lookahead + vitesse*dt
    idx = (progress_idx + np.arange(window)) % n
    dists = np.linalg.norm(path[idx] - pose[:2], axis=1)
    nearest = int(idx[int(np.argmin(dists))])

    # Point d'anticipation : on avance le long du chemin jusqu'a sortir du
    # cercle de rayon lookahead. Arc-length et corde se confondent a 5 cm pres.
    steps_ahead = int(lookahead / PATH_STEP)
    target = path[(nearest + steps_ahead) % n]

    # Transformation dans le repere AGV (rotation -theta) : pure pursuit ne
    # regarde que l'ecart LATERAL y_local du point vise.
    dx, dy = target - pose[:2]
    theta = pose[2]
    y_local = -np.sin(theta) * dx + np.cos(theta) * dy

    # Geometrie pure pursuit : l'arc qui joint l'AGV au point vise a pour
    # courbure kappa = 2*y_local / Ld^2 (corde d'un cercle tangent au cap).
    kappa = 2.0 * y_local / (lookahead ** 2)

    # Ralentissement en virage : a courbure max (coin droit), garder v_nominal
    # saturerait la roue exterieure et ferait deraper la trajectoire.
    v = v_nominal / (1.0 + 1.5 * abs(kappa))
    omega = v * kappa
    return v, omega, nearest


# ---------------------------------------------------------------------------
# 4. Simulation d'un AGV puis d'une mini-flotte
# ---------------------------------------------------------------------------
def simulate_agv(path: np.ndarray, start_idx: int, max_time: float = 90.0) -> dict:
    """Simule un AGV partant de path[start_idx] jusqu'a boucler un tour.

    Retourne les metriques certifiables : erreur laterale max/moyenne,
    distance finale au point de depart, trace complete (pour la flotte).
    """
    n = len(path)
    # Cap initial aligne sur le chemin : un AGV demarre toujours oriente
    # dans son allee (il a ete gare par le shift precedent).
    direction = path[(start_idx + 1) % n] - path[start_idx]
    theta0 = float(np.arctan2(direction[1], direction[0]))
    pose = np.array([path[start_idx, 0], path[start_idx, 1], theta0])

    progress_idx = start_idx
    traveled = 0.0
    perimeter = n * PATH_STEP
    cross_track_errors = []
    trace = [pose[:2].copy()]

    n_ticks = int(max_time / DT)
    for _ in range(n_ticks):
        v, omega, progress_idx = pure_pursuit_step(pose, path, progress_idx, LOOKAHEAD, V_NOMINAL)
        v, omega = saturate_wheels(v, omega)
        pose = integrate_pose(pose, v, omega, DT)
        traveled += abs(v) * DT
        trace.append(pose[:2].copy())

        # Erreur laterale = distance au point du chemin le plus proche (global :
        # ici on VEUT la vraie distance geometrique, pas la fenetre de progression).
        cross_track_errors.append(float(np.min(np.linalg.norm(path - pose[:2], axis=1))))

        # Tour complet : on a parcouru au moins le perimetre ET on est revenu
        # pres du depart (le seul critere distance declencherait au tick 0).
        if traveled >= perimeter * 0.99 and np.linalg.norm(pose[:2] - path[start_idx]) < 0.5:
            break

    errors = np.array(cross_track_errors)
    return {
        "trace": np.array(trace),
        "cte_max": float(errors.max()),
        "cte_mean": float(errors.mean()),
        "final_dist_to_start": float(np.linalg.norm(pose[:2] - path[start_idx])),
        "lap_done": traveled >= perimeter * 0.99,
    }


def simulate_fleet(path: np.ndarray, n_agv: int = 3) -> float:
    """3 AGV decales d'1/3 de boucle : retourne la separation minimale observee.

    Les AGV roulent a la meme vitesse nominale -> l'ecart initial se conserve
    (a la variation pres dans les coins). C'est exactement la strategie de
    slotting temporel utilisee dans FleetSim pour les boucles a sens unique.
    """
    n = len(path)
    offsets = [int(i * n / n_agv) for i in range(n_agv)]
    traces = [simulate_agv(path, off)["trace"] for off in offsets]
    n_ticks = min(len(t) for t in traces)
    min_sep = np.inf
    for a in range(n_agv):
        for b in range(a + 1, n_agv):
            seps = np.linalg.norm(traces[a][:n_ticks] - traces[b][:n_ticks], axis=1)
            min_sep = min(min_sep, float(seps.min()))
    return min_sep


# ---------------------------------------------------------------------------
# Main : verifie chaque critere de reussite par une assertion.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)  # seed fixe : contrainte determinisme LogiSim

    # --- Critere 1 : aller-retour cinematique exact ---
    worst = 0.0
    for _ in range(100):
        v, omega = rng.uniform(-2, 2), rng.uniform(-3, 3)
        v2, omega2 = wheels_to_body(*body_to_wheels(v, omega, WHEEL_BASE), WHEEL_BASE)
        worst = max(worst, abs(v - v2), abs(omega - omega2))
    assert worst < 1e-12, f"roundtrip cinematique: {worst:.2e}"
    print(f"[1] Roundtrip body<->wheels : erreur max {worst:.2e} (< 1e-12)  OK")

    # --- Critere 2 : integration exacte vs Euler ---
    # 2a. Cercle complet en exacte : la pose finale referme le cercle.
    v_c, omega_c = 1.0, 0.5                      # cercle de rayon 2 m
    period = 2 * np.pi / omega_c
    n_steps = int(round(period / 0.05))
    dt_c = period / n_steps                       # dt ajuste pour fermer le tour exactement
    pose_exact = np.array([0.0, 0.0, 0.0])
    for _ in range(n_steps):
        pose_exact = integrate_pose(pose_exact, v_c, omega_c, dt_c)
    err_close = float(np.linalg.norm(pose_exact[:2]))
    assert err_close < 1e-6, f"integration exacte: {err_close:.2e}"

    # 2b. Erreur corde-vs-arc d'Euler sur UN tick grossier (dt=0.5 s, replay de
    # telemetrie 2 Hz) : Euler avance en ligne droite la ou l'AGV decrit un arc
    # serre (R=0.5 m). Note : a (v, omega) constants, Euler trace un polygone
    # regulier FERME — la derive d'Euler ne se voit pas sur l'endpoint d'un
    # cercle complet, elle se voit tick par tick et des que omega varie.
    p_arc = integrate_pose(np.zeros(3), 1.0, 2.0, 0.5)
    p_eul = integrate_pose_euler(np.zeros(3), 1.0, 2.0, 0.5)
    err_euler = float(np.linalg.norm(p_arc[:2] - p_eul[:2]))
    assert err_euler > 0.2, f"Euler devrait s'ecarter de l'arc: {err_euler:.4f}"
    print(f"[2] Cercle complet exact : fermeture {err_close:.2e} m (< 1e-6) ; "
          f"Euler 1 tick a 2 Hz : ecart {err_euler:.3f} m (> 0.2)  OK")

    # --- Criteres 3-4 : suivi de la boucle entrepot ---
    path = warehouse_loop()
    result = simulate_agv(path, start_idx=0)
    assert result["lap_done"], "l'AGV n'a pas boucle son tour"
    assert result["cte_max"] < 0.30, f"cte_max={result['cte_max']:.3f}"
    assert result["cte_mean"] < 0.10, f"cte_mean={result['cte_mean']:.3f}"
    assert result["final_dist_to_start"] < 0.5, f"final={result['final_dist_to_start']:.3f}"
    print(f"[3] Erreur laterale : max {result['cte_max']:.3f} m (< 0.30), "
          f"moyenne {result['cte_mean']:.3f} m (< 0.10)  OK")
    print(f"[4] Tour complet : retour a {result['final_dist_to_start']:.3f} m du depart (< 0.5)  OK")

    # --- Critere 5 : separation de la mini-flotte ---
    min_sep = simulate_fleet(path, n_agv=3)
    assert min_sep > 1.0, f"separation minimale {min_sep:.2f} m"
    print(f"[5] Flotte de 3 AGV : separation minimale {min_sep:.2f} m (> 1.0)  OK")

    # --- Critere 6 : determinisme (meme run -> memes metriques, bit a bit) ---
    result2 = simulate_agv(path, start_idx=0)
    assert result["cte_max"] == result2["cte_max"] and result["cte_mean"] == result2["cte_mean"]
    print("[6] Determinisme : deux runs identiques bit a bit  OK")

    print("\nTous les criteres de reussite passent.")
