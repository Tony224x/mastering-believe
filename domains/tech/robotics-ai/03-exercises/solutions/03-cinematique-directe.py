"""
Solutions des exercices J3 - Cinematique directe.

Couvre :
  - EASY    : FK 3-DOF planaire (cos/sin)
  - MEDIUM  : FK PoE 3D pour bras RRR spherique
  - HARD    : verification + sabotage 1 mm sur Franka Panda (skip si mujoco indisponible)

# requires: numpy, mujoco
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import mujoco  # type: ignore

    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False


# ============================================================================
# Helpers SE(3) reutilisables (dupliques depuis 02-code pour autonomie)
# ============================================================================


def skew(w: Sequence[float]) -> np.ndarray:
    wx, wy, wz = float(w[0]), float(w[1]), float(w[2])
    return np.array([[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]])


def expm_so3(omega: np.ndarray, theta: float) -> np.ndarray:
    if abs(theta) < 1e-12:
        return np.eye(3)
    W = skew(omega)
    return np.eye(3) + np.sin(theta) * W + (1.0 - np.cos(theta)) * (W @ W)


def expm_se3(screw: np.ndarray, theta: float) -> np.ndarray:
    omega = np.asarray(screw[:3], dtype=float)
    v = np.asarray(screw[3:], dtype=float)
    T = np.eye(4)
    nrm = float(np.linalg.norm(omega))
    if nrm < 1e-12:
        T[:3, 3] = v * theta
        return T
    omega_hat = omega / nrm
    v_hat = v / nrm
    th = theta * nrm
    R = expm_so3(omega_hat, th)
    W = skew(omega_hat)
    G = np.eye(3) * th + (1.0 - np.cos(th)) * W + (th - np.sin(th)) * (W @ W)
    T[:3, :3] = R
    T[:3, 3] = G @ v_hat
    return T


def fk_poe(M: np.ndarray, screws: list[np.ndarray], q: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    for S, qi in zip(screws, q):
        T = T @ expm_se3(S, float(qi))
    return T @ M


# ============================================================================
# EASY - FK 3-DOF planaire
# ============================================================================


def fk_3dof_planar(
    theta1: float,
    theta2: float,
    theta3: float,
    L1: float = 0.5,
    L2: float = 0.4,
    L3: float = 0.3,
) -> tuple[float, float, float]:
    """FK d'un bras planaire 3-DOF en serie. Retourne (x, y, phi)."""
    # Angles absolus = somme cumulative des angles relatifs
    a1 = theta1
    a2 = theta1 + theta2
    a3 = theta1 + theta2 + theta3
    x = L1 * np.cos(a1) + L2 * np.cos(a2) + L3 * np.cos(a3)
    y = L1 * np.sin(a1) + L2 * np.sin(a2) + L3 * np.sin(a3)
    return float(x), float(y), float(a3)


def solution_easy() -> None:
    print("=" * 70)
    print("EASY - FK planaire 3-DOF")
    print("=" * 70)
    x, y, phi = fk_3dof_planar(np.deg2rad(45), np.deg2rad(-30), np.deg2rad(60))
    print(f"  effecteur : x={x:.4f}, y={y:.4f}, phi={np.rad2deg(phi):.2f} deg")

    # Verification a la main du x :
    #   a1 = 45, a2 = 15, a3 = 75
    #   x = 0.5*cos(45) + 0.4*cos(15) + 0.3*cos(75)
    #     = 0.5*0.7071 + 0.4*0.9659 + 0.3*0.2588
    #     = 0.3536 + 0.3864 + 0.0776 = 0.8176
    expected_x = (
        0.5 * np.cos(np.deg2rad(45))
        + 0.4 * np.cos(np.deg2rad(15))
        + 0.3 * np.cos(np.deg2rad(75))
    )
    assert abs(x - expected_x) < 1e-9
    print(f"  OK : x verifie a la main = {expected_x:.4f}")
    print()


# ============================================================================
# MEDIUM - FK PoE pour bras RRR 3D (tete spherique simplifiee)
# ============================================================================


def build_spherical_arm() -> tuple[np.ndarray, list[np.ndarray]]:
    """Construit (M, screws) du bras RRR spherique decrit dans l'enonce."""
    # Pose home : effecteur en (0, 0, 1.2), orientation identite
    M = np.eye(4)
    M[:3, 3] = np.array([0.0, 0.0, 1.2])

    # Joint 1 : axe z monde, ancre origine
    omega1 = np.array([0.0, 0.0, 1.0])
    p1 = np.array([0.0, 0.0, 0.0])
    S1 = np.concatenate([omega1, -np.cross(omega1, p1)])

    # Joint 2 : axe y monde, ancre (0, 0, 0.5)
    omega2 = np.array([0.0, 1.0, 0.0])
    p2 = np.array([0.0, 0.0, 0.5])
    S2 = np.concatenate([omega2, -np.cross(omega2, p2)])

    # Joint 3 : axe y monde, ancre (0, 0, 0.9)
    omega3 = np.array([0.0, 1.0, 0.0])
    p3 = np.array([0.0, 0.0, 0.9])
    S3 = np.concatenate([omega3, -np.cross(omega3, p3)])

    return M, [S1, S2, S3]


def solution_medium() -> None:
    print("=" * 70)
    print("MEDIUM - FK PoE 3D bras RRR spherique")
    print("=" * 70)
    M, screws = build_spherical_arm()

    # 1. q = 0 -> on doit retrouver M
    T_a = fk_poe(M, screws, np.zeros(3))
    err_a = np.linalg.norm(T_a - M)
    print(f"  T(q=0) - M Frobenius : {err_a:.2e}")
    assert err_a < 1e-12, "T(0) doit etre exactement M"

    # 2. q_b = (pi/4, pi/6, -pi/3)
    q_b = np.array([np.pi / 4, np.pi / 6, -np.pi / 3])
    T_b = fk_poe(M, screws, q_b)
    np.set_printoptions(precision=4, suppress=True)
    print(f"  T(q_b) =\n{T_b}")

    # 3. Coherence : q = (pi, 0, 0) -> rotation pi autour de z, l'effecteur
    # (sur l'axe z) ne doit PAS bouger en position
    q_c = np.array([np.pi, 0.0, 0.0])
    T_c = fk_poe(M, screws, q_c)
    pos_c = T_c[:3, 3]
    err_c = np.linalg.norm(pos_c - np.array([0, 0, 1.2]))
    print(f"  T(pi,0,0) translation : {pos_c}, ecart vs (0,0,1.2) : {err_c:.2e}")
    assert err_c < 1e-9
    print("  OK : effecteur sur l'axe z, invariant par rotation joint 1.")
    print()


# ============================================================================
# HARD - Panda 7-DOF, verification puis sabotage 1 mm
# ============================================================================


def find_panda_xml() -> Path | None:
    """Localise le MJCF Panda Menagerie, ou None si introuvable."""
    candidates = []
    if "MUJOCO_MENAGERIE" in os.environ:
        candidates.append(Path(os.environ["MUJOCO_MENAGERIE"]) / "franka_emika_panda" / "panda.xml")
    home = Path.home()
    candidates.extend(
        [
            home / "GitRepo" / "mujoco_menagerie" / "franka_emika_panda" / "panda.xml",
            home / "Downloads" / "mujoco_menagerie" / "franka_emika_panda" / "panda.xml",
            Path("./mujoco_menagerie/franka_emika_panda/panda.xml"),
        ]
    )
    for p in candidates:
        if p.exists():
            return p
    return None


def extract_panda_screws_solution(model, data, end_body: str):
    """Version solution : extrait M et screws spatiaux du Panda en config home."""
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, end_body)
    if body_id < 0:
        raise RuntimeError(f"body '{end_body}' introuvable")
    M = np.eye(4)
    M[:3, :3] = data.xmat[body_id].reshape(3, 3).copy()
    M[:3, 3] = data.xpos[body_id].copy()

    screws: list[np.ndarray] = []
    anchors: list[np.ndarray] = []
    omegas: list[np.ndarray] = []
    for j in range(model.njnt):
        if int(model.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            continue
        parent_body = int(model.jnt_bodyid[j])
        R_parent = data.xmat[parent_body].reshape(3, 3)
        omega_local = np.asarray(model.jnt_axis[j], dtype=float)
        omega = R_parent @ omega_local
        omega = omega / np.linalg.norm(omega)
        p = np.asarray(data.xanchor[j], dtype=float)
        S = np.concatenate([omega, -np.cross(omega, p)])
        screws.append(S)
        anchors.append(p.copy())
        omegas.append(omega.copy())
    return M, screws, omegas, anchors


def solution_hard() -> None:
    print("=" * 70)
    print("HARD - Franka Panda : verification + sabotage 1 mm")
    print("=" * 70)

    if not HAS_MUJOCO:
        print("  [SKIP] mujoco non installe.")
        return
    xml = find_panda_xml()
    if xml is None:
        print("  [SKIP] mujoco_menagerie/franka_emika_panda introuvable.")
        return
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)

    end_body = "hand"
    try:
        M, screws, omegas, anchors = extract_panda_screws_solution(model, data, end_body)
    except RuntimeError:
        for cand in ("attachment", "link7", "panda_hand"):
            try:
                M, screws, omegas, anchors = extract_panda_screws_solution(model, data, cand)
                end_body = cand
                break
            except RuntimeError:
                continue
        else:
            print("  [SKIP] aucun body effecteur reconnu")
            return

    n = len(screws)
    print(f"  effecteur : '{end_body}', {n} hinges")

    # Recupere les indices qpos des hinges pour remplir data.qpos correctement
    qpos_addr = [
        int(model.jnt_qposadr[j])
        for j in range(model.njnt)
        if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE)
    ][:n]

    rng = np.random.default_rng(seed=2025)
    qs = rng.uniform(-np.pi / 2, np.pi / 2, size=(20, n))

    def compare(screws_used: list[np.ndarray]) -> tuple[float, float, float, float]:
        """Retourne (mean_frob, max_frob, mean_pos, max_pos)."""
        frobs = []
        pos_errs = []
        for q in qs:
            T_poe = fk_poe(M, screws_used, q)
            mujoco.mj_resetData(model, data)
            for k, qi in zip(qpos_addr, q):
                data.qpos[k] = qi
            mujoco.mj_forward(model, data)
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, end_body)
            T_mj = np.eye(4)
            T_mj[:3, :3] = data.xmat[body_id].reshape(3, 3)
            T_mj[:3, 3] = data.xpos[body_id]
            frobs.append(float(np.linalg.norm(T_poe - T_mj)))
            pos_errs.append(float(np.linalg.norm(T_poe[:3, 3] - T_mj[:3, 3])))
        return (
            float(np.mean(frobs)),
            float(np.max(frobs)),
            float(np.mean(pos_errs)),
            float(np.max(pos_errs)),
        )

    # Phase A
    mean_f, max_f, mean_p, max_p = compare(screws)
    print(f"  Phase A (sain)    : Frob mean={mean_f:.2e} max={max_f:.2e} | pos mean={mean_p:.2e} max={max_p:.2e}")

    # Phase B : sabotage 1 mm sur l'ancre du joint 0 (le plus proximal)
    sabotaged = [s.copy() for s in screws]
    p0 = anchors[0] + np.array([1e-3, 0.0, 0.0])
    sabotaged[0] = np.concatenate([omegas[0], -np.cross(omegas[0], p0)])
    mean_f2, max_f2, mean_p2, max_p2 = compare(sabotaged)
    print(f"  Phase B (1mm bug) : Frob mean={mean_f2:.2e} max={max_f2:.2e} | pos mean={mean_p2:.2e} max={max_p2:.2e}")

    # Phase C : commentaire automatique
    print("  Phase C : analyse")
    print(f"    - le 1 mm proximal produit en moyenne {mean_p2*1000:.2f} mm d'ecart effecteur")
    if mean_p2 > 1.5e-3:
        print("    - amplification > 1 : la chaine cinematique en aval propage l'erreur")
    elif mean_p2 < 5e-4:
        print("    - amplification < 1 : la chaine 'reabsorbe' partiellement l'erreur")
    else:
        print("    - amplification ~ 1 : selon configuration, peut osciller")
    print()


def main() -> None:
    solution_easy()
    solution_medium()
    solution_hard()


if __name__ == "__main__":
    main()
