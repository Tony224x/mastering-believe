"""
J2 — Solutions des exercices easy / medium / hard.

Run : python domains/robotics-ai/03-exercises/solutions/02-transformations-3d.py
"""

# requires: numpy, scipy

import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation as R_scipy


# ==========================================================================
# EASY — rotations elementaires Rx/Ry/Rz a la main + verification SO(3)
# ==========================================================================


def Rx(t: float) -> np.ndarray:
    """Rotation autour de l'axe x (formule a savoir par coeur)."""
    c, s = np.cos(t), np.sin(t)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ]
    )


def Ry(t: float) -> np.ndarray:
    """Rotation autour de l'axe y. Attention au signe du sin."""
    c, s = np.cos(t), np.sin(t)
    # Convention right-handed : Ry inverse les signes par rapport a Rx/Rz
    # parce que y est entre x et z dans la regle de la main droite.
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ]
    )


def Rz(t: float) -> np.ndarray:
    """Rotation autour de l'axe z."""
    c, s = np.cos(t), np.sin(t)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def solve_easy() -> None:
    print("\n--- EASY ---")

    R = Rx(np.deg2rad(30)) @ Ry(np.deg2rad(45)) @ Rz(np.deg2rad(60))

    err_orth = np.max(np.abs(R.T @ R - np.eye(3)))
    det_R = np.linalg.det(R)
    print(f"  max |R^T R - I| = {err_orth:.2e}")
    print(f"  det(R) - 1      = {det_R - 1:.2e}")
    assert err_orth < 1e-10, "R non-orthogonale"
    assert np.isclose(det_R, 1.0, atol=1e-10), "det(R) != +1"

    p = np.array([1.0, 2.0, 3.0])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p

    x = np.array([0.5, 0.0, 0.0])
    x_h = np.append(x, 1.0)  # passage en coord homogenes
    x_transformed = (T @ x_h)[:3]
    # Verification "a la main" : T @ x_h = R @ x + p
    assert np.allclose(x_transformed, R @ x + p)
    print(f"  point transforme = {x_transformed}")
    print("  [OK] easy")


# ==========================================================================
# MEDIUM — composition d'une chaine SE(3) + inverse correct vs faux
# ==========================================================================


def make_T(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def invert_T_correct(T: np.ndarray) -> np.ndarray:
    """L'inverse SE(3) correct : [R^T, -R^T p; 0, 1]."""
    R = T[:3, :3]
    p = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ p
    return out


def invert_T_wrong(T: np.ndarray) -> np.ndarray:
    """L'inverse FAUX : [R^T, -p; 0, 1]. Existe pour pedagogie uniquement."""
    R = T[:3, :3]
    p = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -p  # FAUX : il manque le R^T
    return out


def solve_medium() -> None:
    print("\n--- MEDIUM ---")

    # Construction des trois transformations via scipy (pour montrer l'API).
    R_01 = R_scipy.from_rotvec(np.array([0.0, 0.0, 1.0]) * (np.pi / 4)).as_matrix()
    R_12 = R_scipy.from_rotvec(np.array([0.0, 1.0, 0.0]) * (np.pi / 3)).as_matrix()
    R_23 = np.eye(3)

    T_01 = make_T(R_01, np.array([1.0, 0.0, 0.0]))
    T_12 = make_T(R_12, np.array([0.0, 1.0, 0.0]))
    T_23 = make_T(R_23, np.array([0.0, 0.0, 1.0]))

    T_03 = T_01 @ T_12 @ T_23
    print(f"  T_03 =\n{T_03.round(4)}")

    # Inverse correct vs faux.
    err_correct = np.max(np.abs(T_03 @ invert_T_correct(T_03) - np.eye(4)))
    err_wrong = np.max(np.abs(T_03 @ invert_T_wrong(T_03) - np.eye(4)))
    print(f"  max |T @ T_correct^-1 - I| = {err_correct:.2e}  (doit etre ~0)")
    print(f"  max |T @ T_wrong^-1 - I|   = {err_wrong:.2e}  (doit etre clairement > 0)")
    assert err_correct < 1e-10
    assert err_wrong > 1e-3, "le faux inverse devrait etre clairement faux"

    # Recuperation de T_12 par algebre matricielle :
    #   T_03 = T_01 @ T_12 @ T_23
    #   <=> T_12 = T_01^{-1} @ T_03 @ T_23^{-1}
    T_12_recovered = invert_T_correct(T_01) @ T_03 @ invert_T_correct(T_23)
    err_recover = np.max(np.abs(T_12_recovered - T_12))
    print(f"  max |T_12_recovered - T_12| = {err_recover:.2e}")
    assert err_recover < 1e-10
    print("  [OK] medium")


# ==========================================================================
# HARD — twists, exp_se3, formulation PoE pour bras 2-DOF planaire
# ==========================================================================


def skew(omega: np.ndarray) -> np.ndarray:
    wx, wy, wz = omega
    return np.array(
        [
            [0.0, -wz, wy],
            [wz, 0.0, -wx],
            [-wy, wx, 0.0],
        ]
    )


def twist_to_se3(V: np.ndarray) -> np.ndarray:
    """V = (omega, v) en R^6 -> matrice 4x4 dans se(3)."""
    omega = V[:3]
    v = V[3:]
    M = np.zeros((4, 4))
    M[:3, :3] = skew(omega)
    M[:3, 3] = v
    return M


def exp_se3(V: np.ndarray) -> np.ndarray:
    """Exponentielle matricielle d'un twist : se(3) -> SE(3)."""
    return expm(twist_to_se3(V))


def fk_naive(q1: float, q2: float, L1: float = 1.0, L2: float = 1.0) -> np.ndarray:
    """FK naive du bras 2-DOF planaire (verite terrain)."""
    T_w1 = make_T(Rz(q1), np.zeros(3))
    T_12 = make_T(Rz(q2), np.array([L1, 0.0, 0.0]))
    T_2e = make_T(np.eye(3), np.array([L2, 0.0, 0.0]))
    return T_w1 @ T_12 @ T_2e


def fk_poe(q1: float, q2: float, L1: float = 1.0, L2: float = 1.0) -> np.ndarray:
    """FK via Product of Exponentials.

    Joint 1 : rotation autour de z passant par l'origine.
      omega_1 = (0, 0, 1)
      q_axis_1 = (0, 0, 0)  (origine)
      v_1 = -omega_1 x q_axis_1 = (0, 0, 0)

    Joint 2 : rotation autour de z passant par (L1, 0, 0).
      omega_2 = (0, 0, 1)
      q_axis_2 = (L1, 0, 0)
      v_2 = -omega_2 x q_axis_2 = -((0,0,1) x (L1,0,0)) = -(0, L1, 0) = (0, -L1, 0)

    Le signe moins vient de la formule generale : pour un screw rotoide,
    la composante lineaire spatiale represente la vitesse de l'origine du
    repere espace si on tournait autour de cet axe. Cette origine est
    "tiree" dans la direction opposee au moment, d'ou le moins.
    Reference : Lynch & Park 2017, §3.3.2 et §4.1.2.
    """
    omega_1 = np.array([0.0, 0.0, 1.0])
    v_1 = np.zeros(3)
    S_1 = np.concatenate([omega_1, v_1])

    omega_2 = np.array([0.0, 0.0, 1.0])
    q_axis_2 = np.array([L1, 0.0, 0.0])
    v_2 = -np.cross(omega_2, q_axis_2)
    S_2 = np.concatenate([omega_2, v_2])

    # Configuration home M : end-effector a (L1+L2, 0, 0), orientation = I.
    M = make_T(np.eye(3), np.array([L1 + L2, 0.0, 0.0]))

    return exp_se3(S_1 * q1) @ exp_se3(S_2 * q2) @ M


def solve_hard() -> None:
    print("\n--- HARD ---")

    # Sanite exp_se3 sur 3 cas.
    T_zero = exp_se3(np.zeros(6))
    assert np.allclose(T_zero, np.eye(4), atol=1e-12)

    V_pure_rot = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) * (np.pi / 2)
    T_rot = exp_se3(V_pure_rot)
    expected = make_T(Rz(np.pi / 2), np.zeros(3))
    assert np.allclose(T_rot, expected, atol=1e-10)

    V_pure_trans = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]) * 2.0
    T_trans = exp_se3(V_pure_trans)
    expected = make_T(np.eye(3), np.array([2.0, 0.0, 0.0]))
    assert np.allclose(T_trans, expected, atol=1e-10)
    print("  [OK] sanite exp_se3 (3/3 cas)")

    # Bonus : home config doit donner (2, 0, 0).
    T_home_naive = fk_naive(0.0, 0.0)
    T_home_poe = fk_poe(0.0, 0.0)
    assert np.allclose(T_home_naive, T_home_poe, atol=1e-10)
    assert np.allclose(T_home_naive[:3, 3], [2.0, 0.0, 0.0])
    print("  [OK] home config : PoE = naive = (2, 0, 0)")

    # 5 configurations aleatoires : PoE doit egaler la version naive.
    rng = np.random.default_rng(42)
    qs = rng.uniform(-np.pi, np.pi, size=(5, 2))
    max_err_global = 0.0
    for i, (q1, q2) in enumerate(qs):
        T_n = fk_naive(q1, q2)
        T_p = fk_poe(q1, q2)
        err = float(np.max(np.abs(T_n - T_p)))
        max_err_global = max(max_err_global, err)
        print(f"    q=({q1:+.3f}, {q2:+.3f})  max|naive - PoE| = {err:.2e}")
        assert err < 1e-10, f"PoE mismatch sur config {i}"
    print(f"  [OK] PoE == naive sur 5 configs aleatoires (max_err = {max_err_global:.2e})")


# ==========================================================================
# Main
# ==========================================================================


def main() -> None:
    print("=" * 70)
    print("J2 — Solutions exercices")
    print("=" * 70)
    solve_easy()
    solve_medium()
    solve_hard()
    print("\n" + "=" * 70)
    print("[OK] toutes les solutions passent.")
    print("=" * 70)


if __name__ == "__main__":
    main()
