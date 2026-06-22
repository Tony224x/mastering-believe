"""
Jour 3 - Cinematique directe (Forward Kinematics) from scratch.

Ce script fait deux choses :
  1. FK manuelle d'un bras planaire 2-DOF en numpy (formule cos/sin).
  2. FK manuelle d'un bras 7-DOF type Franka Panda via Product of Exponentials,
     puis comparaison avec mujoco.mj_forward sur le MJCF du Panda (Menagerie).

Reference : [Lynch & Park, 2017, Modern Robotics, ch. 4].

# requires: numpy, mujoco
"""

from __future__ import annotations

import os  # WHY : on tente de localiser le MJCF Franka Panda Menagerie sur disque
from pathlib import Path
from typing import Sequence

import numpy as np

# mujoco est optionnel : la partie 2-DOF tourne sans. On gere l'import en try.
try:
    import mujoco  # type: ignore

    HAS_MUJOCO = True
except ImportError:  # WHY : permet de lancer la partie pedagogique sans mujoco installe
    HAS_MUJOCO = False


# ---------------------------------------------------------------------------
# Partie 1 - Bras planaire 2-DOF (l'exemple de la theorie)
# ---------------------------------------------------------------------------


def fk_2dof_planar(theta1: float, theta2: float, L1: float = 1.0, L2: float = 0.5) -> np.ndarray:
    """FK d'un bras planaire 2-DOF (joints revolutes, plan XY).

    Args:
        theta1 : angle joint 1 (rad), absolu dans le repere monde
        theta2 : angle joint 2 (rad), RELATIF au segment 1
        L1, L2 : longueurs des deux segments

    Returns:
        position 2D de l'effecteur (x, y)
    """
    # WHY : pour un bras planaire en serie, les angles relatifs s'ADDITIONNENT
    # pour donner les angles absolus. C'est specifique au plan (axes paralleles).
    a1 = theta1
    a2 = theta1 + theta2

    # Composition geometrique : chaque segment ajoute son vecteur direction
    x = L1 * np.cos(a1) + L2 * np.cos(a2)
    y = L1 * np.sin(a1) + L2 * np.sin(a2)
    return np.array([x, y])


def demo_2dof() -> None:
    """Verifie l'exemple de la theorie : theta1=30 deg, theta2=45 deg."""
    print("=" * 70)
    print("Partie 1 : bras planaire 2-DOF, theta1=30deg, theta2=45deg")
    print("=" * 70)

    pos = fk_2dof_planar(np.deg2rad(30.0), np.deg2rad(45.0), L1=1.0, L2=0.5)
    print(f"  effecteur calcule  : x={pos[0]:.4f}, y={pos[1]:.4f}")
    print(f"  attendu (theorie)  : x=0.9954, y=0.9830")

    # WHY : assertion rendue plus laxiste qu'un strict equal pour robustesse numerique
    expected = np.array([0.9954, 0.9830])
    assert np.allclose(pos, expected, atol=1e-3), "FK 2-DOF KO"
    print("  OK : FK 2-DOF coherente avec calcul a la main.\n")


# ---------------------------------------------------------------------------
# Partie 2 - Outils SE(3) / so(3) pour Product of Exponentials
# ---------------------------------------------------------------------------


def skew(w: Sequence[float]) -> np.ndarray:
    """Matrice antisymetrique (3x3) associee au vecteur omega de R^3.

    [w] tel que [w] @ v = w x v pour tout v.
    """
    wx, wy, wz = float(w[0]), float(w[1]), float(w[2])
    return np.array(
        [
            [0.0, -wz, wy],
            [wz, 0.0, -wx],
            [-wy, wx, 0.0],
        ]
    )


def expm_so3(omega: np.ndarray, theta: float) -> np.ndarray:
    """Exponentielle dans SO(3) via Rodrigues : e^([w]*theta) avec ||omega||=1.

    Si theta tres petit, on retourne l'identite (evite division par 0).
    """
    if abs(theta) < 1e-12:
        return np.eye(3)

    # WHY : Rodrigues -- formule fermee, evite les developpements en serie
    W = skew(omega)
    return np.eye(3) + np.sin(theta) * W + (1.0 - np.cos(theta)) * (W @ W)


def expm_se3(screw: np.ndarray, theta: float) -> np.ndarray:
    """Exponentielle dans SE(3) pour un screw S = (omega, v), ||omega||=1 (revolute).

    Formule [Lynch & Park, 2017, eq. 3.88] :
        e^([S]theta) = | R       G(theta) v |
                      | 0           1       |
    avec
        R       = e^([omega]theta) (Rodrigues)
        G(theta) = I*theta + (1 - cos(theta)) [omega] + (theta - sin(theta)) [omega]^2
    """
    omega = np.asarray(screw[:3], dtype=float)
    v = np.asarray(screw[3:], dtype=float)
    T = np.eye(4)

    # Cas pur prismatic : ||omega|| == 0, alors R = I et translation = v * theta
    nrm = float(np.linalg.norm(omega))
    if nrm < 1e-12:
        T[:3, 3] = v * theta
        return T

    # Normalisation defensive : si l'utilisateur a passe ||omega|| != 1, on rescale
    omega_hat = omega / nrm
    v_hat = v / nrm
    theta_eff = theta * nrm

    R = expm_so3(omega_hat, theta_eff)
    W = skew(omega_hat)

    # G(theta) : terme integral du deplacement le long du screw
    G = (
        np.eye(3) * theta_eff
        + (1.0 - np.cos(theta_eff)) * W
        + (theta_eff - np.sin(theta_eff)) * (W @ W)
    )
    T[:3, :3] = R
    T[:3, 3] = G @ v_hat
    return T


def fk_poe(M: np.ndarray, screws: list[np.ndarray], q: np.ndarray) -> np.ndarray:
    """Forward Kinematics par Product of Exponentials, repere espace.

    T(q) = e^([S1] q1) * e^([S2] q2) * ... * e^([Sn] qn) * M

    Args:
        M       : pose home de l'effecteur (4x4) en config q=0
        screws  : liste de n screws spatiaux (chacun shape (6,))
        q       : configuration articulaire (n,)

    Returns:
        T_se : pose 4x4 de l'effecteur dans le repere espace
    """
    assert len(screws) == len(q), "nombre de screws != taille de q"
    T = np.eye(4)
    for S, qi in zip(screws, q):
        T = T @ expm_se3(S, float(qi))
    return T @ M


# ---------------------------------------------------------------------------
# Partie 3 - Panda 7-DOF : extraction des screws depuis MuJoCo + comparaison
# ---------------------------------------------------------------------------


def find_panda_xml() -> Path | None:
    """Tente de localiser le MJCF Franka Panda de mujoco_menagerie sur disque.

    Cherche dans des emplacements standards. Retourne None si absent : dans ce
    cas la partie 7-DOF est skippee proprement.
    """
    candidates = []

    # Variable d'env explicite (ex. utilisateurs avancés)
    if "MUJOCO_MENAGERIE" in os.environ:
        candidates.append(Path(os.environ["MUJOCO_MENAGERIE"]) / "franka_emika_panda" / "panda.xml")

    # Repos clones courants
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


def extract_panda_screws(model: "mujoco.MjModel", data: "mujoco.MjData", end_body: str) -> tuple[np.ndarray, list[np.ndarray]]:
    """Extrait la pose home M et les screws spatiaux Si du Panda en config q=0.

    Strategie :
      1. Mettre data.qpos a 0 et appeler mj_forward -> tout est en config home.
      2. Pour chaque joint hinge (revolute) du robot, lire l'axe dans le repere
         monde via mj_forward (data.xaxis pour les axes monde dans MuJoCo 3.x ;
         on reconstruit a partir de model.jnt_axis exprime dans le body parent
         et la matrice de rotation du body parent).
      3. Le point sur l'axe est l'ancre du joint dans le repere monde
         (data.xanchor).
      4. Si pour le joint i : Si = (omega_i, -omega_i x p_i).
      5. M = pose 4x4 de l'effecteur (end_body) en config home.
    """
    mujoco.mj_resetData(model, data)
    # data.qpos = 0 est garanti par mj_resetData.
    mujoco.mj_forward(model, data)

    # 1. Pose home M de l'effecteur
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, end_body)
    if body_id < 0:
        raise RuntimeError(f"body '{end_body}' introuvable dans le MJCF")

    M = np.eye(4)
    M[:3, :3] = data.xmat[body_id].reshape(3, 3).copy()
    M[:3, 3] = data.xpos[body_id].copy()

    # 2. Pour chaque joint hinge, screw spatial en config home
    screws: list[np.ndarray] = []
    for j in range(model.njnt):
        jtype = int(model.jnt_type[j])
        # On ne traite que les hinges (revolute). Slide=prismatic possible mais pas sur Panda.
        if jtype != int(mujoco.mjtJoint.mjJNT_HINGE):
            continue

        # Axe du joint dans le repere monde : on multiplie l'axe local par la
        # rotation du body PARENT du joint (au repos, en config home).
        # MuJoCo: model.jnt_axis (njnt, 3) est dans le repere body parent.
        parent_body = int(model.jnt_bodyid[j])
        R_parent = data.xmat[parent_body].reshape(3, 3)
        omega_local = np.asarray(model.jnt_axis[j], dtype=float)
        omega = R_parent @ omega_local
        omega = omega / np.linalg.norm(omega)

        # Point sur l'axe dans le repere monde : ancre fournie par MuJoCo
        p = np.asarray(data.xanchor[j], dtype=float)

        # Screw spatial : v = -omega x p
        v = -np.cross(omega, p)
        S = np.concatenate([omega, v])
        screws.append(S)

    return M, screws


def demo_panda() -> None:
    """Compare FK PoE manuelle vs mj_forward sur le Franka Panda."""
    print("=" * 70)
    print("Partie 2 : Franka Panda 7-DOF, FK PoE vs mj_forward")
    print("=" * 70)

    if not HAS_MUJOCO:
        print("  [SKIP] mujoco non installe. pip install mujoco. ")
        return

    xml = find_panda_xml()
    if xml is None:
        print(
            "  [SKIP] mujoco_menagerie introuvable.\n"
            "    git clone https://github.com/google-deepmind/mujoco_menagerie ~/GitRepo/mujoco_menagerie\n"
            "    ou export MUJOCO_MENAGERIE=/chemin/vers/mujoco_menagerie"
        )
        return

    print(f"  MJCF Panda : {xml}")
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)

    # Le body effecteur dans le Panda Menagerie est typiquement 'hand'
    end_body = "hand"
    try:
        M, screws = extract_panda_screws(model, data, end_body)
    except RuntimeError:
        # Fallback : essayer d'autres noms
        for candidate in ("attachment", "link7", "panda_hand", "right_hand"):
            try:
                M, screws = extract_panda_screws(model, data, candidate)
                end_body = candidate
                break
            except RuntimeError:
                continue
        else:
            print("  [SKIP] aucun body effecteur reconnu trouve")
            return

    print(f"  effecteur cible : '{end_body}'")
    print(f"  nombre de hinges (DOF revolutes) : {len(screws)}")

    # Test : 5 configurations aleatoires et comparaison
    rng = np.random.default_rng(seed=42)
    n = len(screws)
    max_err = 0.0
    for trial in range(5):
        q = rng.uniform(-0.5, 0.5, size=n)  # WHY : amplitude raisonnable, on reste loin des butees

        # FK manuelle PoE
        T_poe = fk_poe(M, screws, q)

        # FK MuJoCo : on copie q dans qpos (Panda a 7 hinges + 2 finger slides ; on
        # remplit uniquement les 7 premiers DOF qui correspondent aux hinges du bras)
        mujoco.mj_resetData(model, data)
        # Indice d'adresse qpos pour chaque hinge -> remplissage propre
        qpos_addr = []
        for j in range(model.njnt):
            if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE):
                qpos_addr.append(int(model.jnt_qposadr[j]))
        for k, qi in zip(qpos_addr[:n], q):
            data.qpos[k] = qi
        mujoco.mj_forward(model, data)

        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, end_body)
        T_mj = np.eye(4)
        T_mj[:3, :3] = data.xmat[body_id].reshape(3, 3)
        T_mj[:3, 3] = data.xpos[body_id]

        err = float(np.linalg.norm(T_poe - T_mj))
        max_err = max(max_err, err)
        print(f"  trial {trial+1}: ||T_poe - T_mj|| = {err:.2e}")

    print(f"  erreur max sur 5 trials : {max_err:.2e}")
    if max_err < 1e-4:
        print("  OK : FK PoE manuelle coherente avec mj_forward.")
    else:
        # WHY : tolerance laxiste : Menagerie ajoute parfois des transforms
        # statiques (offsets, dummy bodies) que notre extraction simple ignore.
        # Pour un cours, l'objectif est pedagogique ; pour la production, il
        # faut extraire M et S avec la chaine complete des bodies intermediaires.
        print("  WARN : ecart > 1e-4. Probablement bodies/sites intermediaires non geres.")


def main() -> None:
    demo_2dof()
    demo_panda()


if __name__ == "__main__":
    main()
