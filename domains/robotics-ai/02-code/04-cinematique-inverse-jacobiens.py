"""
J4 - Cinematique inverse + Jacobiens
=====================================

Demos progressives :
1. IK analytique pour un bras 2-DOF planaire (closed-form).
2. Calcul du Jacobien : analytique vs numerique (difference finies).
3. IK numerique avec damped least squares (DLS).
4. Detection et gestion d'une singularite.
5. Bras redondant 3-DOF planaire : pseudo-inverse + objectif secondaire.

Source : [Lynch & Park, 2017, ch. 5-6], CS223A Khatib L6-L8.

# requires: numpy, scipy
"""

import numpy as np

# scipy.optimize est dans les dependances pour les variantes "IK comme probleme
# d'optimisation contrainte" (limites articulaires, evitement obstacles).
# On l'importe mollement : la demo ci-dessous tourne en numpy pur, mais l'import
# documente la deps et permet d'enrichir l'exo HARD avec scipy.optimize.minimize.
try:
    from scipy.optimize import minimize  # noqa: F401
except ImportError:  # pragma: no cover - fallback pedagogique
    minimize = None

# Reproductibilite des restarts aleatoires
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# 1. Bras 2-DOF planaire : FK et IK analytique
# ---------------------------------------------------------------------------

# Longueurs des deux segments. On garde L1=L2=1 pour simplifier la geometrie.
L1, L2 = 1.0, 1.0


def fk_2dof(q):
    """Forward kinematics du bras 2-DOF planaire.

    q = [q1, q2] en radians.
    Retourne la position (x, y) de l'end-effector.
    """
    q1, q2 = q
    # Somme des deux segments : segment 1 part de l'origine, segment 2 part du coude.
    x = L1 * np.cos(q1) + L2 * np.cos(q1 + q2)
    y = L1 * np.sin(q1) + L2 * np.sin(q1 + q2)
    return np.array([x, y])


def ik_2dof_analytique(target, elbow="up"):
    """IK closed-form pour le 2-DOF planaire.

    target : (x, y) cible.
    elbow  : "up" (q2 > 0) ou "down" (q2 < 0). C'est la multiplicite classique.
    Retourne (q1, q2) ou leve ValueError si hors workspace.
    """
    x, y = target
    # Loi des cosinus : on calcule q2 a partir de la distance carree a l'origine.
    cos_q2 = (x ** 2 + y ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    # Si |cos_q2| > 1, la cible est hors du workspace (anneau [|L1-L2|, L1+L2]).
    if abs(cos_q2) > 1.0:
        raise ValueError(f"Cible {target} hors workspace (cos_q2={cos_q2:.3f})")
    # Deux signes possibles : c'est exactement la multiplicite coude-haut/coude-bas.
    sign = 1.0 if elbow == "up" else -1.0
    q2 = sign * np.arccos(np.clip(cos_q2, -1.0, 1.0))
    # On utilise atan2 (et non atan) pour gerer les 4 quadrants sans ambiguite.
    q1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    return np.array([q1, q2])


# ---------------------------------------------------------------------------
# 2. Jacobien : analytique vs numerique
# ---------------------------------------------------------------------------


def jacobian_2dof_analytique(q):
    """Jacobien analytique du 2-DOF planaire (derivees calculees a la main)."""
    q1, q2 = q
    s1, c1 = np.sin(q1), np.cos(q1)
    s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)
    # J[i, j] = d(x_i) / d(q_j). Voir derivation manuelle dans la theorie.
    return np.array(
        [
            [-L1 * s1 - L2 * s12, -L2 * s12],
            [L1 * c1 + L2 * c12, L2 * c12],
        ]
    )


def jacobian_numerique(fk_fn, q, eps=1e-6):
    """Jacobien par differences finies centrees.

    Generique : marche pour n'importe quelle FK qu'on lui passe.
    Strategie : df/dq_j ~ (f(q + eps e_j) - f(q - eps e_j)) / (2 eps).

    O(n) evaluations FK. Suffisant pour n petit, mais pour Franka 7-DOF en
    boucle de controle on prefere la version analytique (ou autodiff via
    JAX/PyTorch) pour la perf.
    """
    n = len(q)
    f0 = fk_fn(q)
    m = len(f0)
    J = np.zeros((m, n))
    for j in range(n):
        dq = np.zeros(n)
        dq[j] = eps
        # Difference finie centree : meilleure precision que one-sided pour le meme cout.
        J[:, j] = (fk_fn(q + dq) - fk_fn(q - dq)) / (2 * eps)
    return J


# ---------------------------------------------------------------------------
# 3. IK numerique avec damped least squares (DLS)
# ---------------------------------------------------------------------------


def ik_dls(
    fk_fn,
    jac_fn,
    target,
    q_init,
    *,
    lam=0.05,
    tol=1e-4,
    max_iter=200,
    step_clip=0.5,
    q_min=None,
    q_max=None,
    verbose=False,
):
    """IK numerique generique avec damped least squares.

    fk_fn(q) -> pose. jac_fn(q) -> J. target = pose desiree.

    DLS : dq = J^T (J J^T + lam^2 I)^{-1} e
    Equivaut a regulariser ||J dq - e||^2 + lam^2 ||dq||^2 (Tikhonov).
    Empeche l'explosion de la pseudo-inverse pres des singularites.

    step_clip : borne sur ||dq|| pour eviter les sauts dans les zones non lineaires.
    q_min/q_max : limites articulaires (clamp simple a chaque iteration).

    Retourne (q_solution, infos).
    """
    q = np.array(q_init, dtype=float)
    m = len(fk_fn(q))  # dimension de la pose
    I = np.eye(m)
    history = []

    for k in range(max_iter):
        e = target - fk_fn(q)
        err_norm = np.linalg.norm(e)
        history.append(err_norm)
        if verbose:
            print(f"  iter {k:3d}  ||e|| = {err_norm:.6e}")
        if err_norm < tol:
            return q, {"converged": True, "iters": k, "history": history}

        J = jac_fn(q)
        # DLS step : on resout (J J^T + lam^2 I) y = e puis dq = J^T y.
        # np.linalg.solve est plus stable et plus rapide que np.linalg.inv.
        try:
            y = np.linalg.solve(J @ J.T + lam ** 2 * I, e)
        except np.linalg.LinAlgError:
            # Si meme avec la regularisation ca casse, on abandonne proprement.
            return q, {"converged": False, "iters": k, "history": history, "reason": "linalg"}
        dq = J.T @ y

        # Step-size cap : on borne ||dq|| pour rester dans la zone de validite de la linearisation.
        norm_dq = np.linalg.norm(dq)
        if norm_dq > step_clip:
            dq = dq * (step_clip / norm_dq)

        q = q + dq
        # Clamp aux limites articulaires si fournies.
        if q_min is not None and q_max is not None:
            q = np.clip(q, q_min, q_max)

    return q, {"converged": False, "iters": max_iter, "history": history, "reason": "max_iter"}


# ---------------------------------------------------------------------------
# 4. Detection de singularite (manipulability)
# ---------------------------------------------------------------------------


def manipulability(J):
    """Indice de manipulabilite de Yoshikawa : sqrt(det(J J^T)).

    = 0 exactement en singularite (J perd du rang).
    On peut aussi regarder la plus petite valeur singuliere : meme info, plus
    robuste numeriquement quand det est petit.
    """
    # On utilise SVD : plus stable que det quand le conditionnement est mauvais.
    sv = np.linalg.svd(J, compute_uv=False)
    # Pour J m x n avec m <= n : sqrt(det(J J^T)) = produit des valeurs singulieres.
    return float(np.prod(sv))


def smallest_singular_value(J):
    """Plus petite valeur singuliere de J. Indicateur direct de proximite singularite."""
    return float(np.linalg.svd(J, compute_uv=False)[-1])


# ---------------------------------------------------------------------------
# 5. Bras 3-DOF planaire (redondant pour une cible 2D)
# ---------------------------------------------------------------------------

L = (1.0, 1.0, 0.5)  # longueurs des trois segments


def fk_3dof(q):
    """FK du bras 3-DOF planaire."""
    q1, q2, q3 = q
    x = L[0] * np.cos(q1) + L[1] * np.cos(q1 + q2) + L[2] * np.cos(q1 + q2 + q3)
    y = L[0] * np.sin(q1) + L[1] * np.sin(q1 + q2) + L[2] * np.sin(q1 + q2 + q3)
    return np.array([x, y])


def jacobian_3dof(q):
    """Jacobien analytique 2x3 du 3-DOF planaire (redondance dim 1)."""
    q1, q2, q3 = q
    s1, c1 = np.sin(q1), np.cos(q1)
    s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)
    s123, c123 = np.sin(q1 + q2 + q3), np.cos(q1 + q2 + q3)
    return np.array(
        [
            [-L[0] * s1 - L[1] * s12 - L[2] * s123, -L[1] * s12 - L[2] * s123, -L[2] * s123],
            [L[0] * c1 + L[1] * c12 + L[2] * c123, L[1] * c12 + L[2] * c123, L[2] * c123],
        ]
    )


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------


def demo_1_analytique():
    print("=" * 70)
    print("Demo 1 : IK analytique 2-DOF, multiplicite coude-haut / coude-bas")
    print("=" * 70)
    target = np.array([1.2, 0.6])
    for elbow in ("up", "down"):
        q = ik_2dof_analytique(target, elbow=elbow)
        x_check = fk_2dof(q)
        # Sanity check : on ferme la boucle FK(IK(target)) et on doit retomber sur target.
        print(f"  elbow={elbow:4s}  q={q}  FK(q)={x_check}  err={np.linalg.norm(x_check - target):.2e}")
    print()


def demo_2_jacobien():
    print("=" * 70)
    print("Demo 2 : Jacobien analytique vs numerique (difference finies)")
    print("=" * 70)
    q = np.array([0.3, -0.5])
    J_ana = jacobian_2dof_analytique(q)
    J_num = jacobian_numerique(fk_2dof, q)
    # Les deux doivent etre identiques au bruit numerique pres (~1e-10).
    print("  J analytique :")
    print(J_ana)
    print("  J numerique  :")
    print(J_num)
    print(f"  ||J_ana - J_num|| = {np.linalg.norm(J_ana - J_num):.2e}")
    print()


def demo_3_dls():
    print("=" * 70)
    print("Demo 3 : IK numerique avec DLS sur le 2-DOF")
    print("=" * 70)
    target = np.array([1.2, 0.6])
    q_init = np.array([0.1, 0.1])  # initialisation generique
    q, info = ik_dls(fk_2dof, jacobian_2dof_analytique, target, q_init, lam=0.05, verbose=False)
    print(f"  converged = {info['converged']} en {info['iters']} iterations")
    print(f"  q_DLS     = {q}")
    print(f"  q_ana(up) = {ik_2dof_analytique(target, 'up')}  (l'une des deux solutions)")
    print(f"  err pose  = {np.linalg.norm(target - fk_2dof(q)):.2e}")
    print()


def demo_4_singularite():
    print("=" * 70)
    print("Demo 4 : pres d'une singularite (bras tendu : q2 ~ 0)")
    print("=" * 70)
    # Bras quasi tendu : la matrice J devient mal conditionnee.
    q_sing = np.array([0.5, 1e-3])
    J_sing = jacobian_2dof_analytique(q_sing)
    print(f"  q                 = {q_sing}")
    print(f"  manipulability    = {manipulability(J_sing):.4e}")
    print(f"  sigma_min(J)      = {smallest_singular_value(J_sing):.4e}")
    print(f"  cond(J)           = {np.linalg.cond(J_sing):.2e}")

    # On essaie d'atteindre une cible tout au bord du workspace : (L1+L2, 0) = (2, 0).
    target = np.array([2.0 - 1e-4, 0.0])

    print("\n  Pseudo-inverse pure (lam=0) :")
    q1_res, info1 = ik_dls(
        fk_2dof, jacobian_2dof_analytique, target, q_init=np.array([0.5, 0.5]), lam=0.0, max_iter=50
    )
    print(f"    converged={info1['converged']} en {info1['iters']} iter, ||e||={info1['history'][-1]:.2e}")

    print("  DLS (lam=0.1) :")
    q2_res, info2 = ik_dls(
        fk_2dof, jacobian_2dof_analytique, target, q_init=np.array([0.5, 0.5]), lam=0.1, max_iter=50
    )
    print(f"    converged={info2['converged']} en {info2['iters']} iter, ||e||={info2['history'][-1]:.2e}")
    # Le commentaire pedagogique : sans amortissement, l'iteration peut osciller / exploser.
    # Avec DLS, on perd un peu en precision finale (||e|| > tol potentiellement) mais on reste stable.
    print()


def demo_5_redondant():
    print("=" * 70)
    print("Demo 5 : bras 3-DOF redondant (cible 2D, 3 articulations)")
    print("=" * 70)
    target = np.array([1.5, 0.5])
    # Deux initialisations differentes -> deux solutions differentes (parmi une famille a 1 parametre).
    for q_init in (np.array([0.1, 0.1, 0.1]), np.array([0.5, -0.5, 0.5])):
        q, info = ik_dls(fk_3dof, jacobian_3dof, target, q_init, lam=0.05)
        print(f"  init={q_init}  ->  q*={q}  pose={fk_3dof(q)}  iters={info['iters']}")
    # Le noyau de J (dim 1 ici) materialise la redondance : on peut bouger les articulations
    # sans bouger l'effecteur. C'est ce qu'on exploite pour des objectifs secondaires
    # (ex. maximiser la manipulabilite, eviter les limites, contourner un obstacle).
    print()


def demo_6_force_statique():
    print("=" * 70)
    print("Demo 6 : forces statiques tau = J^T F (dualite cinematique/statique)")
    print("=" * 70)
    q = np.array([0.4, 0.6])
    J = jacobian_2dof_analytique(q)
    # On veut que l'effecteur exerce une force horizontale F = (5 N, 0) sur l'environnement.
    F = np.array([5.0, 0.0])
    tau = J.T @ F
    print(f"  q   = {q}")
    print(f"  F   = {F} (force cartesienne sur l'environnement)")
    print(f"  tau = J^T F = {tau} (couples articulaires necessaires en N.m)")
    print()


if __name__ == "__main__":
    demo_1_analytique()
    demo_2_jacobien()
    demo_3_dls()
    demo_4_singularite()
    demo_5_redondant()
    demo_6_force_statique()
