"""
J7 — Solutions des exercices Easy/Medium/Hard.

Each exercise function is independent. Run all : `python 07-perception-3d-robotique.py`.

Sources :
    - [Open3D docs] open3d.org pour les references conceptuelles.
    - Procrustes / Kabsch algorithm (Besl & McKay 1992) pour ICP closed-form.

# requires: numpy, scipy
"""

import numpy as np
from scipy.spatial import cKDTree


# ============================================================================
# Easy : projection pinhole reciproque
# ============================================================================

def exercice_easy() -> None:
    print("\n=== EASY : projection pinhole + back-projection ===")
    # 1. Matrice intrinseque pour 640x480 avec focale 500 px et principal point centre.
    K = np.array([
        [500.0,   0.0, 320.0],
        [  0.0, 500.0, 240.0],
        [  0.0,   0.0,   1.0],
    ])
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 2. Cinq points 3D arbitraires dans le repere camera (Z>0 obligatoirement).
    points = np.array([
        [0.0, 0.0, 2.0],     # exactement sur l'axe optique a 2m
        [0.5, 0.3, 1.5],     # decale lateralement
        [-0.2, -0.1, 1.0],   # decale dans l'autre sens
        [0.1, -0.2, 3.0],    # plus loin
        [-0.4, 0.2, 2.5],
    ])

    # 3. Projection : u = fx * X/Z + cx, v = fy * Y/Z + cy.
    u = fx * (points[:, 0] / points[:, 2]) + cx
    v = fy * (points[:, 1] / points[:, 2]) + cy
    depths = points[:, 2]
    print("Projection (u, v, depth) :")
    for i in range(5):
        print(f"  point {i} : ({u[i]:.2f}, {v[i]:.2f}, {depths[i]:.2f})")

    # Le premier point sur l'axe optique tombe pile sur (cx, cy) = (320, 240).
    assert np.isclose(u[0], cx) and np.isclose(v[0], cy), \
        "Un point sur l'axe optique doit se projeter sur le principal point"

    # 4. Inverse : (u, v, depth) -> (X, Y, Z).
    X = (u - cx) * depths / fx
    Y = (v - cy) * depths / fy
    Z = depths
    reconstructed = np.stack([X, Y, Z], axis=1)

    # 5. Verification reciproque
    assert np.allclose(reconstructed, points, atol=1e-9), \
        "La back-projection doit retrouver les points originaux"
    print("Back-projection reciproque : OK (np.allclose passe).")


# ============================================================================
# Medium : voxelisation manuelle + normales
# ============================================================================

def voxelize(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Voxelisation : 1 point par cube (centroide des points dans ce cube).

    Vectorise via np.unique(return_inverse=True) puis np.add.at.
    """
    # Index entier du voxel de chaque point. floor + division est equivalent
    # a un binning regulier sur les 3 axes.
    indices = np.floor(points / voxel_size).astype(np.int64)

    # On a besoin d'un identifiant unique par voxel. On peut utiliser le tuple
    # (ix, iy, iz) — np.unique accepte un axis=0 sur tableau 2D.
    unique_voxels, inverse = np.unique(indices, axis=0, return_inverse=True)

    # Calcule centroide par voxel : somme des points puis divise par count.
    n_voxels = unique_voxels.shape[0]
    sums = np.zeros((n_voxels, 3))
    counts = np.zeros(n_voxels)
    np.add.at(sums, inverse, points)
    np.add.at(counts, inverse, 1)
    return sums / counts[:, None]


def estimate_normals(points: np.ndarray, k: int = 15) -> np.ndarray:
    """Pour chaque point, normale = vecteur propre de la covariance locale
    associe a la plus petite valeur propre. KD-tree pour les voisins.
    """
    tree = cKDTree(points)
    # Cherche k voisins (le point lui-meme inclus, donc on voudra k+1 puis [:, 1:]).
    _, idx = tree.query(points, k=k)
    normals = np.zeros_like(points)
    for i, neigh_idx in enumerate(idx):
        neigh = points[neigh_idx]
        cov = np.cov(neigh.T)  # 3x3
        eigvals, eigvecs = np.linalg.eigh(cov)  # eigh trie par ordre croissant
        normals[i] = eigvecs[:, 0]  # plus petite valeur propre = normale
    return normals


def exercice_medium() -> None:
    print("\n=== MEDIUM : voxelisation + normales ===")
    # Sphere uniforme rayon 0.5
    rng = np.random.default_rng(0)
    N = 20000
    phi = rng.uniform(0, 2 * np.pi, N)
    theta = np.arccos(rng.uniform(-1, 1, N))
    r = 0.5
    sphere = np.stack([
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta),
    ], axis=1)

    voxelized = voxelize(sphere, voxel_size=0.05)
    print(f"Sphere : {sphere.shape[0]} -> voxelise : {voxelized.shape[0]} points")
    assert 200 <= voxelized.shape[0] <= 1000, \
        f"Le nombre de voxels {voxelized.shape[0]} est hors de la fourchette attendue"

    # Normales — sur une sphere, doit s'aligner avec le rayon (signe arbitraire
    # car la PCA n'oriente pas la normale).
    normals = estimate_normals(voxelized, k=15)
    radii = voxelized / np.linalg.norm(voxelized, axis=1, keepdims=True)
    alignment = np.abs(np.einsum("ij,ij->i", normals, radii))
    print(f"Alignement moyen normale/rayon : {alignment.mean():.4f}")
    assert alignment.mean() > 0.95, \
        "L'alignement moyen doit etre > 0.95 sur une sphere"


# ============================================================================
# Hard : ICP from scratch + rayon de convergence
# ============================================================================

def icp(source: np.ndarray, target: np.ndarray, max_iter: int = 30,
        tol: float = 1e-7) -> tuple[np.ndarray, list[float]]:
    """ICP point-to-point. Retourne T (4x4) et l'historique de RMS."""
    T = np.eye(4)
    src = source.copy()
    history: list[float] = []
    tree = cKDTree(target)
    prev_rmse = np.inf

    for _ in range(max_iter):
        # Voisin le plus proche (KD-tree, beaucoup plus rapide qu'une matrice
        # complete pour N grand).
        dists, nn_idx = tree.query(src, k=1)
        nn = target[nn_idx]

        # Procrustes / Kabsch
        mu_src = src.mean(axis=0)
        mu_nn = nn.mean(axis=0)
        H = (src - mu_src).T @ (nn - mu_nn)
        U, _, Vt = np.linalg.svd(H)
        # Correction du determinant pour rester dans SO(3) (det = +1)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
        t = mu_nn - R @ mu_src

        # Compose
        T_iter = np.eye(4)
        T_iter[:3, :3] = R
        T_iter[:3, 3] = t
        T = T_iter @ T
        src = (R @ src.T).T + t

        rmse = float(np.sqrt(np.mean(dists ** 2)))
        history.append(rmse)
        if abs(prev_rmse - rmse) < tol:
            break
        prev_rmse = rmse

    return T, history


def make_test_cloud(n: int = 500, seed: int = 1) -> np.ndarray:
    """Cube creux pour tester ICP."""
    rng = np.random.default_rng(seed)
    pts_per_face = n // 6
    faces = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            uv = rng.uniform(-0.5, 0.5, (pts_per_face, 2))
            face = np.zeros((pts_per_face, 3))
            free = [a for a in range(3) if a != axis]
            face[:, free[0]] = uv[:, 0]
            face[:, free[1]] = uv[:, 1]
            face[:, axis] = 0.5 * sign
            faces.append(face)
    return np.vstack(faces)


def rot_z(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def angle_from_R(R: np.ndarray) -> float:
    """Angle de la rotation (radians) extrait de la trace."""
    cos_angle = (np.trace(R) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.arccos(cos_angle))


def exercice_hard() -> None:
    print("\n=== HARD : rayon de convergence ICP ===")
    src = make_test_cloud(n=600)

    print(f"{'theta_input':>12} | {'theta_recov':>12} | {'rmse_final':>12} | {'iters':>5}")
    print("-" * 60)
    for theta_deg in [5, 15, 30, 60, 90, 120, 180]:
        theta = np.deg2rad(theta_deg)
        target = (rot_z(theta) @ src.T).T
        T_est, hist = icp(src, target, max_iter=50)
        theta_recov = np.rad2deg(angle_from_R(T_est[:3, :3]))
        print(f"{theta_deg:>12} | {theta_recov:>12.2f} | {hist[-1]:>12.4f} | {len(hist):>5}")

    # Multi-init pour le cas difficile theta=120 deg.
    print("\n-- Strategie multi-init pour theta = 120 deg --")
    theta = np.deg2rad(120.0)
    target = (rot_z(theta) @ src.T).T
    best_T, best_rmse = None, np.inf
    for init_deg in range(0, 360, 30):
        T_init = np.eye(4)
        T_init[:3, :3] = rot_z(np.deg2rad(init_deg))
        src_init = (T_init[:3, :3] @ src.T).T
        T_est, hist = icp(src_init, target, max_iter=30)
        if hist[-1] < best_rmse:
            best_rmse = hist[-1]
            # Compose avec l'init pour ramener dans le repere src d'origine
            T_full = T_est @ T_init
            best_T = T_full
    theta_recov = np.rad2deg(angle_from_R(best_T[:3, :3]))
    print(f"Meilleur rmse (multi-init) : {best_rmse:.4f}, "
          f"theta retrouve : {theta_recov:.2f} deg (verite 120)")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    exercice_easy()
    exercice_medium()
    exercice_hard()


if __name__ == "__main__":
    main()
