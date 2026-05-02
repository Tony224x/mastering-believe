"""
J2 — Transformations 3D : SE(3), rotations, twists.

Demontre :
  1. Construction et validation d'une matrice de rotation R (R^T R = I, det = +1)
  2. Conversion quaternion <-> matrice de rotation (via scipy.spatial.transform)
  3. Composition de matrices homogenes 4x4 le long d'une chaine cinematique
     (bras planaire 2-DOF utilise comme fil rouge dans la theorie)
  4. Inverse correct d'une transformation SE(3) : T^{-1} = [R^T, -R^T p; 0, 1]
  5. Twist (omega, v) -> exp([V]) -> matrice SE(3) (formule Rodrigues 6D)

Sources : [Lynch & Park, 2017, ch. 3], [Khatib CS223A, L2-3].

Run : python domains/robotics-ai/02-code/02-transformations-3d.py
"""

# requires: numpy, scipy

import numpy as np
from scipy.spatial.transform import Rotation as R_scipy


# --------------------------------------------------------------------------
# 1. Matrices de rotation SO(3)
# --------------------------------------------------------------------------

def rotz(theta: float) -> np.ndarray:
    """Rotation autour de l'axe z d'angle theta (radians).

    On utilise la formule explicite plutot qu'un Rodrigues general parce que
    rotz est ce qui apparait dans tous les bras planaires : un seul DOF, axe
    de rotation aligne avec z monde.
    """
    c, s = np.cos(theta), np.sin(theta)
    # Construction directe : la 3eme ligne/colonne reste l'identite
    # parce qu'on ne touche pas a z.
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def is_rotation_matrix(R: np.ndarray, atol: float = 1e-8) -> bool:
    """Verifie que R est dans SO(3) : R^T R = I ET det(R) = +1.

    Le second test exclut les reflexions, qui satisfont aussi R^T R = I mais
    avec det = -1. C'est l'erreur classique : valider l'orthogonalite seule
    laisse passer des matrices qui inversent l'orientation.
    """
    if R.shape != (3, 3):
        return False
    orthogonal = np.allclose(R.T @ R, np.eye(3), atol=atol)
    proper = np.isclose(np.linalg.det(R), 1.0, atol=atol)
    return orthogonal and proper


# --------------------------------------------------------------------------
# 2. Conversion quaternion <-> matrice de rotation
# --------------------------------------------------------------------------

def demo_quaternion_roundtrip() -> None:
    """Cycle complet : axe-angle -> quaternion -> matrice -> quaternion.

    On verifie qu'on retombe sur le quaternion d'origine (a un signe pres,
    puisque q et -q representent la meme rotation : c'est le double-cover
    SU(2) -> SO(3)).
    """
    # Axe-angle : 90 degres autour de l'axe (1, 1, 1)/sqrt(3).
    axis = np.array([1.0, 1.0, 1.0])
    axis = axis / np.linalg.norm(axis)  # important : axe doit etre unitaire
    angle = np.pi / 2

    # Construction du quaternion via scipy. Convention scipy : (x, y, z, w)
    # — w est le scalaire EN DERNIER. C'est l'inverse de la convention Hamilton
    # (w, x, y, z) utilisee dans la theorie. Erreur de convention = bug de signe.
    rotvec = axis * angle
    q_initial = R_scipy.from_rotvec(rotvec).as_quat()  # (x, y, z, w)

    # Quaternion -> matrice
    R_matrix = R_scipy.from_quat(q_initial).as_matrix()
    assert is_rotation_matrix(R_matrix), "R issue de quaternion doit etre dans SO(3)"

    # Matrice -> quaternion
    q_recovered = R_scipy.from_matrix(R_matrix).as_quat()

    # Le quaternion doit etre egal a +/- q_initial. On compare en valeur absolue
    # apres alignement de signe (dot positif).
    if np.dot(q_initial, q_recovered) < 0:
        q_recovered = -q_recovered
    assert np.allclose(q_initial, q_recovered, atol=1e-10), "round-trip casse"

    print("  [OK] axe-angle -> quat -> R -> quat retombe sur le quaternion d'origine")
    print(f"       R^T R - I max abs = {np.max(np.abs(R_matrix.T @ R_matrix - np.eye(3))):.2e}")
    print(f"       det(R) - 1        = {np.linalg.det(R_matrix) - 1:.2e}")


# --------------------------------------------------------------------------
# 3. Matrices homogenes SE(3) et composition
# --------------------------------------------------------------------------

def make_T(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Construit la matrice homogene 4x4 a partir de R (3x3) et p (3,).

    Layout :
        [ R   p ]
        [ 0   1 ]

    On insiste sur la copie : si R ou p sont modifies plus tard, T ne doit
    pas changer (sinon une chaine cinematique partagerait des references).
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    """Inverse correct d'une transformation SE(3) : T^{-1} = [R^T, -R^T p; 0, 1].

    Erreur classique a NE PAS commettre : retourner [R^T, -p] (la translation
    doit etre re-exprimee dans le nouveau repere via -R^T p).
    """
    R = T[:3, :3]
    p = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ p
    return T_inv


def fk_planar_2dof(q1: float, q2: float, L1: float = 1.0, L2: float = 1.0) -> np.ndarray:
    """Cinematique directe d'un bras planaire 2-DOF (l'exemple fil rouge).

    Chaine : monde {w} -> {1} -> {2} -> {e} (end-effector).
      - T_w1(q1) : rotation autour de z d'angle q1, translation nulle (joint 1
        a l'origine).
      - T_12(q2) : rotation autour de z d'angle q2, translation (L1, 0, 0)
        (le joint 2 est au bout du lien 1).
      - T_2e     : translation pure (L2, 0, 0) (l'end-effector au bout du
        lien 2), rotation = identite.

    A q1 = q2 = 0 on doit retomber sur (L1 + L2, 0, 0) — c'est le test de
    sanite que la theorie utilise.
    """
    T_w1 = make_T(rotz(q1), np.zeros(3))
    T_12 = make_T(rotz(q2), np.array([L1, 0.0, 0.0]))
    T_2e = make_T(np.eye(3), np.array([L2, 0.0, 0.0]))

    # Composition : on lit de gauche a droite le long de la chaine.
    # numpy operator @ = produit matriciel (PEP 465).
    T_we = T_w1 @ T_12 @ T_2e
    return T_we


# --------------------------------------------------------------------------
# 4. Twists et exponentielle matricielle se(3) -> SE(3)
# --------------------------------------------------------------------------

def skew(omega: np.ndarray) -> np.ndarray:
    """Matrice antisymetrique (skew) telle que skew(omega) @ v = omega x v.

    C'est l'objet [omega] dans la notation Lynch & Park.
    """
    wx, wy, wz = omega
    return np.array(
        [
            [0.0, -wz, wy],
            [wz, 0.0, -wx],
            [-wy, wx, 0.0],
        ]
    )


def twist_to_se3(V: np.ndarray) -> np.ndarray:
    """Forme matricielle 4x4 d'un twist V = (omega, v) en R^6.

    Layout (Lynch & Park §3.3.2) :
        [ [omega]  v ]
        [    0    0 ]
    """
    assert V.shape == (6,), "twist doit etre 6D : (omega_x, omega_y, omega_z, v_x, v_y, v_z)"
    omega = V[:3]
    v = V[3:]
    M = np.zeros((4, 4))
    M[:3, :3] = skew(omega)
    M[:3, 3] = v
    # La 4eme ligne reste tout zero — c'est l'algebre de Lie se(3),
    # pas SE(3) (qui aurait un 1 en position [3, 3]).
    return M


def exp_se3(V: np.ndarray) -> np.ndarray:
    """Exponentielle matricielle d'un twist : se(3) -> SE(3).

    On s'appuie sur scipy.linalg.expm pour la formule fermee. Lynch & Park
    eq. 3.88 donne aussi une forme close en termes de Rodrigues 3D + un terme
    G(theta) ; ici on prefere expm pour la lisibilite (le but est pedagogique,
    pas la performance).
    """
    from scipy.linalg import expm  # import local : dependance optionnelle

    return expm(twist_to_se3(V))


# --------------------------------------------------------------------------
# Demos / verifications
# --------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("J2 — Transformations 3D : SE(3), rotations, twists")
    print("=" * 70)

    print("\n[1] Validation de R = rotz(pi/4)")
    R = rotz(np.pi / 4)
    assert is_rotation_matrix(R), "rotz(pi/4) doit etre dans SO(3)"
    print(f"  R^T R close a I  : {np.allclose(R.T @ R, np.eye(3))}")
    print(f"  det(R)           : {np.linalg.det(R):.6f} (doit etre +1)")

    print("\n[2] Round-trip quaternion <-> matrice")
    demo_quaternion_roundtrip()

    print("\n[3] Composition d'une chaine cinematique 2-DOF")
    # Test 1 : configuration home, doit donner end-effector a (L1+L2, 0, 0).
    T_home = fk_planar_2dof(q1=0.0, q2=0.0, L1=1.0, L2=1.0)
    p_home = T_home[:3, 3]
    print(f"  q=(0, 0)             -> p = {p_home}  (attendu [2, 0, 0])")
    assert np.allclose(p_home, [2.0, 0.0, 0.0]), "FK home casse"

    # Test 2 : q1 = pi/2, q2 = 0 — bras pointe en y, position (0, L1+L2).
    T_quart = fk_planar_2dof(q1=np.pi / 2, q2=0.0)
    p_quart = T_quart[:3, 3]
    print(f"  q=(pi/2, 0)          -> p = {p_quart}  (attendu [0, 2, 0])")
    assert np.allclose(p_quart, [0.0, 2.0, 0.0], atol=1e-10), "FK pi/2 casse"

    # Test 3 : q1 = 0, q2 = pi/2 — le 2eme lien fait un coude a 90 deg.
    # Position attendue : (L1, L2, 0) = (1, 1, 0).
    T_elbow = fk_planar_2dof(q1=0.0, q2=np.pi / 2)
    p_elbow = T_elbow[:3, 3]
    print(f"  q=(0, pi/2)          -> p = {p_elbow}  (attendu [1, 1, 0])")
    assert np.allclose(p_elbow, [1.0, 1.0, 0.0], atol=1e-10), "FK coude casse"

    print("\n[4] Inverse correct d'une SE(3)")
    T = T_elbow
    T_inv = invert_T(T)
    # Verification : T @ T_inv = I.
    should_be_I = T @ T_inv
    err = np.max(np.abs(should_be_I - np.eye(4)))
    print(f"  max | T @ T^-1 - I | = {err:.2e}  (doit etre ~0)")
    assert err < 1e-10, "inverse SE(3) casse"

    # Comparaison avec l'inverse FAUX (juste -p au lieu de -R^T p) pour
    # montrer que la difference est non-triviale.
    T_inv_wrong = np.eye(4)
    T_inv_wrong[:3, :3] = T[:3, :3].T
    T_inv_wrong[:3, 3] = -T[:3, 3]  # FAUX
    err_wrong = np.max(np.abs(T @ T_inv_wrong - np.eye(4)))
    print(f"  max | T @ T_wrong - I | = {err_wrong:.2e}  (illustre l'erreur classique)")

    print("\n[5] Twist -> SE(3) via exponentielle")
    # Twist pur rotation : omega = (0, 0, 1), v = 0, parametre theta = pi/2.
    # On doit retomber sur rotz(pi/2) en 4x4.
    theta = np.pi / 2
    V_rot = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) * theta
    T_rot = exp_se3(V_rot)
    expected_rot = make_T(rotz(theta), np.zeros(3))
    print(f"  exp(twist rotation)  matches rotz(pi/2)  : {np.allclose(T_rot, expected_rot, atol=1e-10)}")

    # Twist pur translation : omega = 0, v = (1, 0, 0), parametre = 2.
    # On doit retomber sur translation pure (2, 0, 0).
    V_trans = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]) * 2.0
    T_trans = exp_se3(V_trans)
    expected_trans = make_T(np.eye(3), np.array([2.0, 0.0, 0.0]))
    print(f"  exp(twist translation) matches T(2, 0, 0): {np.allclose(T_trans, expected_trans, atol=1e-10)}")

    # Twist mixte (screw avec pitch) : rotation autour de z + translation le
    # long de z. Pour theta = pi/2, omega = (0, 0, 1), v = (0, 0, 1) on
    # devrait obtenir rotz(pi/2) ET p_z = pi/2.
    V_screw = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0]) * (np.pi / 2)
    T_screw = exp_se3(V_screw)
    print(f"  exp(twist screw mixte) ->")
    print(f"    R = {T_screw[:3, :3].round(3).tolist()}")
    print(f"    p = {T_screw[:3, 3].round(6).tolist()}  (attendu p_z = pi/2 ~ 1.5708)")
    assert np.isclose(T_screw[2, 3], np.pi / 2, atol=1e-10)

    print("\n" + "=" * 70)
    print("[OK] toutes les verifications passent.")
    print("=" * 70)


if __name__ == "__main__":
    main()
