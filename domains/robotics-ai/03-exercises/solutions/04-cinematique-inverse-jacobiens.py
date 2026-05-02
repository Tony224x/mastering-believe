"""
J4 - Solutions des exercices : cinematique inverse + Jacobiens
==============================================================

EASY   : IK analytique 2-DOF + sanity check FK(IK(target)).
MEDIUM : IK numerique DLS sur 3-DOF planaire, comparaison lam in {0, 0.05, 0.5}.
HARD   : IK redondant avec objectif secondaire (null-space projection).

# requires: numpy
"""

import numpy as np


# ===========================================================================
# EASY
# ===========================================================================


def fk_2dof(q, L1=1.0, L2=1.0):
    """FK 2-DOF planaire."""
    q1, q2 = q
    x = L1 * np.cos(q1) + L2 * np.cos(q1 + q2)
    y = L1 * np.sin(q1) + L2 * np.sin(q1 + q2)
    return np.array([x, y])


def ik_2dof(target, L1=1.0, L2=1.0, elbow="up"):
    """IK closed-form 2-DOF planaire. Retourne (q1, q2). Leve ValueError hors workspace."""
    x, y = target
    cos_q2 = (x ** 2 + y ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    if abs(cos_q2) > 1.0 + 1e-12:
        # Hors workspace : pas de solution reelle.
        raise ValueError(f"Cible {target} hors workspace (cos_q2={cos_q2:.4f})")
    # Clip pour eviter NaN sur les bords numeriques (cos_q2 = 1.0000000003 par ex.).
    cos_q2 = np.clip(cos_q2, -1.0, 1.0)
    sign = 1.0 if elbow == "up" else -1.0
    q2 = sign * np.arccos(cos_q2)
    q1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    return np.array([q1, q2])


def solution_easy():
    print("=" * 60)
    print("EASY : IK analytique 2-DOF")
    print("=" * 60)
    targets_in = [(1.5, 0.5), (2.0, 0.0), (-1.0, 0.5)]
    for target in targets_in:
        target = np.array(target)
        print(f"\n  cible = {target}")
        for elbow in ("up", "down"):
            try:
                q = ik_2dof(target, elbow=elbow)
                err = np.linalg.norm(fk_2dof(q) - target)
                print(f"    elbow={elbow:4s}  q={q}  ||FK(q)-target||={err:.2e}")
                assert err < 1e-10, "Erreur FK(IK) trop grande"
            except ValueError as e:
                print(f"    elbow={elbow:4s}  ValueError : {e}")

    # Cible hors workspace : on s'attend a un ValueError.
    print(f"\n  cible = (3.0, 0.0)  (hors workspace)")
    try:
        ik_2dof((3.0, 0.0))
        print("    ECHEC : aurait du lever ValueError")
    except ValueError as e:
        print(f"    ValueError correct : {e}")
    print()


# ===========================================================================
# MEDIUM : IK DLS sur 3-DOF planaire
# ===========================================================================


L = (1.0, 1.0, 0.5)


def fk_3dof(q):
    q1, q2, q3 = q
    x = L[0] * np.cos(q1) + L[1] * np.cos(q1 + q2) + L[2] * np.cos(q1 + q2 + q3)
    y = L[0] * np.sin(q1) + L[1] * np.sin(q1 + q2) + L[2] * np.sin(q1 + q2 + q3)
    return np.array([x, y])


def jac_3dof(q):
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


def ik_dls(fk_fn, jac_fn, target, q_init, lam=0.05, tol=1e-4, max_iter=200, step_clip=None):
    """IK numerique avec damped least squares."""
    q = np.array(q_init, dtype=float)
    m = len(fk_fn(q))
    I = np.eye(m)
    history = []

    for k in range(max_iter):
        e = target - fk_fn(q)
        history.append(np.linalg.norm(e))
        if history[-1] < tol:
            return q, {"converged": True, "iters": k, "history": history}

        J = jac_fn(q)
        # DLS step : gere le cas singulier sans exploser.
        try:
            y = np.linalg.solve(J @ J.T + lam ** 2 * I, e)
        except np.linalg.LinAlgError:
            return q, {"converged": False, "iters": k, "history": history, "reason": "linalg"}
        dq = J.T @ y

        if step_clip is not None:
            n = np.linalg.norm(dq)
            if n > step_clip:
                dq = dq * (step_clip / n)

        q = q + dq

    return q, {"converged": False, "iters": max_iter, "history": history, "reason": "max_iter"}


def solution_medium():
    print("=" * 60)
    print("MEDIUM : IK DLS 3-DOF, comparaison lam in {0, 0.05, 0.5}")
    print("=" * 60)
    target = np.array([1.8, 0.4])
    q_init_normal = np.array([0.1, 0.1, 0.1])
    q_init_singular = np.array([0.0, 1e-3, 1e-3])

    print("\n  Init normal :")
    for lam in (0.0, 0.05, 0.5):
        q_star, info = ik_dls(fk_3dof, jac_3dof, target, q_init_normal, lam=lam)
        last_err = info["history"][-1]
        print(
            f"    lam={lam:.2f}  conv={info['converged']!s:5s}  iters={info['iters']:3d}  "
            f"||e_final||={last_err:.2e}  q*={q_star}"
        )

    print("\n  Init proche singularite (bras tendu) :")
    for lam in (0.0, 0.05, 0.5):
        q_star, info = ik_dls(fk_3dof, jac_3dof, target, q_init_singular, lam=lam, max_iter=300)
        last_err = info["history"][-1]
        print(
            f"    lam={lam:.2f}  conv={info['converged']!s:5s}  iters={info['iters']:3d}  "
            f"||e_final||={last_err:.2e}"
        )
    # Lecture : avec lam=0, la pseudo-inverse pure peut soit converger vite (si chance), soit
    # diverger / tourner en rond pres d'une singularite. Avec DLS, on perd un peu en vitesse
    # de convergence finale mais on est stable et robuste.
    print()


# ===========================================================================
# HARD : Null-space projection avec objectif secondaire
# ===========================================================================


def manipulability(J):
    sv = np.linalg.svd(J, compute_uv=False)
    return float(np.prod(sv))


def grad_manipulability_numeric(jac_fn, q, eps=1e-5):
    """Gradient numerique de w(q) = sqrt(det(J J^T)) par differences centrees."""
    n = len(q)
    g = np.zeros(n)
    for i in range(n):
        dq = np.zeros(n)
        dq[i] = eps
        w_plus = manipulability(jac_fn(q + dq))
        w_minus = manipulability(jac_fn(q - dq))
        g[i] = (w_plus - w_minus) / (2 * eps)
    return g


def ik_dls_redundant(
    fk_fn,
    jac_fn,
    target,
    q_init,
    *,
    q_dot_sec_fn=None,
    lam=0.05,
    tol=1e-4,
    max_iter=200,
    step_clip=0.3,
):
    """IK DLS avec projection dans le null-space pour un objectif secondaire."""
    q = np.array(q_init, dtype=float)
    n = len(q)
    m = len(fk_fn(q))
    I_m = np.eye(m)
    I_n = np.eye(n)
    history = []

    for k in range(max_iter):
        e = target - fk_fn(q)
        history.append(np.linalg.norm(e))
        if history[-1] < tol and q_dot_sec_fn is None:
            # Si pas d'objectif secondaire, on s'arrete des qu'on a converge.
            return q, {"converged": True, "iters": k, "history": history}

        J = jac_fn(q)
        # Pseudo-inverse amortie (gauche : J^T (J J^T + lam^2 I)^{-1}).
        J_pinv = J.T @ np.linalg.inv(J @ J.T + lam ** 2 * I_m)
        dq_primary = J_pinv @ e

        if q_dot_sec_fn is not None:
            dq_sec_full = q_dot_sec_fn(q)
            # Projection dans le noyau de J : (I - J^+ J) tue toute composante "primaire".
            null_proj = I_n - J_pinv @ J
            dq_sec = null_proj @ dq_sec_full
        else:
            dq_sec = np.zeros(n)

        dq = dq_primary + dq_sec

        # Clip global pour rester dans la zone de validite de la linearisation.
        norm_dq = np.linalg.norm(dq)
        if norm_dq > step_clip:
            dq = dq * (step_clip / norm_dq)

        q = q + dq

        # Avec objectif secondaire, on continue d'iterer meme apres avoir atteint la tolerance
        # primaire, pour laisser le secondaire optimiser dans le null-space. On s'arrete si
        # le mouvement total devient minuscule.
        if history[-1] < tol and np.linalg.norm(dq) < 1e-5:
            return q, {"converged": True, "iters": k, "history": history}

    return q, {"converged": False, "iters": max_iter, "history": history}


def solution_hard():
    print("=" * 60)
    print("HARD : IK redondant avec objectif secondaire null-space")
    print("=" * 60)

    target = np.array([1.5, 0.0])
    q_init = np.array([1.5, -1.0, 1.5])  # init excentree, pour bien voir l'effet du secondaire
    q_min = np.array([-np.pi, -np.pi, -np.pi])
    q_max = np.array([np.pi, np.pi, np.pi])
    q_center = (q_min + q_max) / 2  # = 0 ici

    # Objectif (a) : tirer vers le centre des limites articulaires.
    def grad_center(q, k=2.0):
        return -k * (q - q_center)

    print("\n  (a) sans objectif secondaire :")
    q_a0, _ = ik_dls_redundant(fk_3dof, jac_3dof, target, q_init, q_dot_sec_fn=None)
    print(f"    q*  = {q_a0}")
    print(f"    pose= {fk_3dof(q_a0)}  ||e||={np.linalg.norm(target - fk_3dof(q_a0)):.2e}")
    print(f"    ||q*-q_center||={np.linalg.norm(q_a0 - q_center):.3f}")

    print("\n  (a) avec objectif secondaire 'aller vers le centre des limites' :")
    q_a, _ = ik_dls_redundant(fk_3dof, jac_3dof, target, q_init, q_dot_sec_fn=grad_center)
    print(f"    q*  = {q_a}")
    print(f"    pose= {fk_3dof(q_a)}  ||e||={np.linalg.norm(target - fk_3dof(q_a)):.2e}")
    print(f"    ||q*-q_center||={np.linalg.norm(q_a - q_center):.3f}  <-- doit etre plus petit")

    # Objectif (b) : maximiser la manipulabilite (s'eloigner des singularites).
    def grad_manip(q, alpha=5.0):
        return alpha * grad_manipulability_numeric(jac_3dof, q)

    print("\n  (b) avec objectif secondaire 'maximiser la manipulabilite' :")
    q_b, _ = ik_dls_redundant(fk_3dof, jac_3dof, target, q_init, q_dot_sec_fn=grad_manip)
    print(f"    q*  = {q_b}")
    print(f"    pose= {fk_3dof(q_b)}  ||e||={np.linalg.norm(target - fk_3dof(q_b)):.2e}")
    print(f"    manipulability(q*) = {manipulability(jac_3dof(q_b)):.4f}")
    print(f"    manipulability(sans secondaire) = {manipulability(jac_3dof(q_a0)):.4f}")

    # Sanity check : le projecteur null-space est de rang n - m = 1 pour un 3-DOF en 2D.
    # Important : on utilise la *vraie* pseudo-inverse (np.linalg.pinv), pas la version
    # amortie. Le projecteur orthogonal sur null(J) est P = I - pinv(J) @ J ; il est
    # idempotent et de rang exactement n - m. La pseudo-inverse amortie (lam > 0) brise
    # cette structure en echange de stabilite numerique pres des singularites.
    J = jac_3dof(q_a)
    J_pinv = np.linalg.pinv(J)
    P = np.eye(3) - J_pinv @ J
    print(f"\n  Verifications structurelles (avec vraie pseudo-inverse) :")
    print(f"    rang du projecteur null-space = {np.linalg.matrix_rank(P, tol=1e-6)}  (attendu : 1)")
    print(f"    ||P @ P - P|| (idempotence)   = {np.linalg.norm(P @ P - P):.2e}  (~ 0 attendu)")
    # Verif additionnelle : tout vecteur dans le null-space ne doit pas bouger l'effecteur.
    null_vec = P @ np.array([1.0, 1.0, 1.0])
    print(f"    ||J @ null_vec||              = {np.linalg.norm(J @ null_vec):.2e}  (~ 0 attendu)")
    print()


if __name__ == "__main__":
    solution_easy()
    solution_medium()
    solution_hard()
