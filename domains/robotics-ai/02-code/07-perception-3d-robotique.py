"""
J7 — Perception 3D pour robotique : pinhole, point clouds, ICP.

Ce script fait trois choses, dans l'ordre :
    1. Genere un point cloud synthetique (cube creux) — pas besoin de capteur.
    2. Simule une camera pinhole : prend un point cloud 3D et le projette en
       image (u, v, depth) ; puis l'inverse (image -> point cloud) pour
       montrer que le pipeline RGB-D est juste de l'algebre.
    3. Aligne deux nuages avec ICP point-to-point. Si Open3D est dispo on
       utilise sa version optimisee ; sinon on tombe sur un ICP from scratch
       en numpy pur (tres lisible, ~30 lignes).

Sources :
    - [Open3D docs] open3d.org pour les briques registration et geometry.
    - [V-JEPA 2, Meta 2025] cite en theorie comme l'horizon (representation
      latente vs reconstruction explicite). Ici on reste dans l'explicite.

# requires: numpy, open3d (optional — fallback numpy si absent)
"""

# Imports standard. numpy est obligatoire — toute la geometrie 3D passe par lui.
import numpy as np

# Open3D est optionnel : si absent on bascule sur un ICP en numpy pur. On
# encode ca par un flag HAS_O3D pour que le reste du fichier reste lisible.
try:
    import open3d as o3d  # type: ignore
    HAS_O3D = True
except ImportError:
    o3d = None  # placeholder pour mypy / lecteurs
    HAS_O3D = False


# ============================================================================
# 1. Generation d'un point cloud synthetique : surface d'un cube
# ============================================================================

def make_cube_point_cloud(side: float = 1.0, points_per_face: int = 200,
                          seed: int = 0) -> np.ndarray:
    """Genere ~6 * points_per_face points uniformement sur les 6 faces d'un cube.

    On evite un cube *plein* parce qu'un capteur RGB-D ne voit que la surface
    (la profondeur s'arrete au premier obstacle). Un point cloud de surface
    est donc plus realiste pour tester ICP.
    """
    # RNG local pour la reproductibilite — c'est plus propre que np.random.seed
    # qui touche un etat global partage.
    rng = np.random.default_rng(seed)
    half = side / 2.0
    faces = []  # accumule les points face par face

    # Pour chaque axe (0=x, 1=y, 2=z), on genere les deux faces opposees
    # (axis = +half et axis = -half). Les deux autres coordonnees sont
    # tirees uniformement dans [-half, +half]. On obtient une surface plane.
    for axis in range(3):
        for sign in (-1.0, 1.0):
            uv = rng.uniform(-half, half, size=(points_per_face, 2))
            face = np.zeros((points_per_face, 3))
            # Place les deux coordonnees libres
            free_axes = [a for a in range(3) if a != axis]
            face[:, free_axes[0]] = uv[:, 0]
            face[:, free_axes[1]] = uv[:, 1]
            # Et fixe la coordonnee de la face (constante pour cette face)
            face[:, axis] = sign * half
            faces.append(face)

    # Concatene en un seul tableau (N, 3). Ordre quelconque — un point cloud
    # est non ordonne par definition.
    return np.vstack(faces)


# ============================================================================
# 2. Camera pinhole : 3D -> pixel et pixel -> 3D
# ============================================================================

def project_pinhole(points_camera: np.ndarray, K: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Projette un point cloud (N, 3) du repere camera vers l'image.

    Conventions : axe Z de la camera = direction du regard (vers la scene).
    Les points avec Z<=0 sont *derriere* la camera et inobservables.

    Returns:
        uv     : (N_visible, 2) coordonnees pixel des points visibles
        depths : (N_visible,)   profondeurs Z correspondantes
    """
    # Filtre les points derriere la camera. Sans ca on aurait des projections
    # absurdes (division par Z<=0).
    in_front = points_camera[:, 2] > 1e-6
    pts = points_camera[in_front]

    # u = fx * X/Z + cx, v = fy * Y/Z + cy. Vectorise sur l'axe 0.
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * (pts[:, 0] / pts[:, 2]) + cx
    v = fy * (pts[:, 1] / pts[:, 2]) + cy

    uv = np.stack([u, v], axis=1)
    depths = pts[:, 2]
    return uv, depths


def unproject_depth_image(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Inverse de la projection : depth map -> point cloud dans repere camera.

    `depth` est un (H, W) ; un pixel a 0 signifie "pas de retour" (sera ignore).
    """
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Grille de coordonnees pixel. meshgrid avec indexing='xy' donne
    # u (largeur, axe 1) et v (hauteur, axe 0) dans le bon ordre.
    u, v = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")

    # Inversion par formule. d=0 sera filtre apres.
    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    # On jette les pixels invalides — un capteur reel a souvent des trous
    # (reflections speculaires, hors portee).
    pts = pts[pts[:, 2] > 1e-6]
    return pts


# ============================================================================
# 3. ICP point-to-point — version numpy from scratch
# ============================================================================

def icp_numpy(source: np.ndarray, target: np.ndarray, max_iter: int = 30,
              tol: float = 1e-6) -> tuple[np.ndarray, float]:
    """ICP point-to-point en numpy pur.

    Trouve T ∈ SE(3) (4x4) qui minimise sum_i ||T(source_i) - nearest(target)||^2.
    Utilise SVD pour la solution closed-form de Procrustes a chaque iteration
    (Besl & McKay 1992). Pas optimise — pour pedagogie.

    Returns:
        T    : matrice homogene (4, 4)
        rmse : erreur RMS finale entre source aligne et target
    """
    # Initialise T comme l'identite. Une vraie pipeline injecterait une init
    # via global registration (FPFH+RANSAC) — la on assume une init proche.
    T = np.eye(4)
    src = source.copy()  # on transforme une copie a chaque iter
    prev_rmse = np.inf

    for _ in range(max_iter):
        # 1. Pour chaque point source, trouver son voisin le plus proche
        #    dans target. O(N*M) — pour des nuages > 5k points il faudrait
        #    un KD-tree. Ici on reste vectorise et explicite.
        # diffs[i, j] = src[i] - target[j], shape (N, M, 3)
        diffs = src[:, None, :] - target[None, :, :]
        d2 = np.sum(diffs * diffs, axis=-1)  # distances au carre (N, M)
        nn_idx = np.argmin(d2, axis=1)
        nn = target[nn_idx]  # (N, 3) le voisin de chaque src

        # 2. Procrustes : aligner src sur nn (deux nuages de meme taille,
        #    en correspondance ordonnee maintenant).
        # Centre les deux ensembles
        mu_src = src.mean(axis=0)
        mu_nn = nn.mean(axis=0)
        src_c = src - mu_src
        nn_c = nn - mu_nn

        # H = src_c^T @ nn_c. SVD de H.
        H = src_c.T @ nn_c
        U, _, Vt = np.linalg.svd(H)
        # Correction du determinant pour eviter une reflexion (det=-1)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1.0, 1.0, d])
        R = Vt.T @ D @ U.T
        t = mu_nn - R @ mu_src

        # 3. Compose la transformation iterative dans T global.
        T_iter = np.eye(4)
        T_iter[:3, :3] = R
        T_iter[:3, 3] = t
        T = T_iter @ T

        # Applique a la copie source pour l'iteration suivante
        src = (R @ src.T).T + t

        # Critere d'arret
        rmse = float(np.sqrt(np.mean(np.sum((src - nn) ** 2, axis=1))))
        if abs(prev_rmse - rmse) < tol:
            break
        prev_rmse = rmse

    return T, rmse


# ============================================================================
# 4. Demo principale
# ============================================================================

def main() -> None:
    print(f"[J7] Open3D dispo : {HAS_O3D}")

    # ---- 1. Genere un point cloud cube et le tourne / translate -----------
    src = make_cube_point_cloud(side=1.0, points_per_face=150, seed=42)
    print(f"[J7] Point cloud genere : {src.shape[0]} points sur les 6 faces")

    # Construit une transformation de verite-terrain (rotation 15 deg autour
    # de Z + translation) qu'on appliquera pour creer la cible. ICP devra
    # retrouver l'inverse de cette transformation.
    theta = np.deg2rad(15.0)
    R_gt = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0],
    ])
    t_gt = np.array([0.10, 0.05, 0.02])
    target = (R_gt @ src.T).T + t_gt
    print(f"[J7] Verite-terrain : rotation 15deg autour Z + translation {t_gt}")

    # ---- 2. Simule la camera pinhole sur la source -----------------------
    # Place la camera reculee de 3 m sur l'axe Z, regardant le cube.
    # Pour ca on translate les points : repere camera = repere monde decale.
    K = np.array([
        [500.0,   0.0, 320.0],
        [  0.0, 500.0, 240.0],
        [  0.0,   0.0,   1.0],
    ])
    # Decale le cube dans la direction +Z (devant la camera) de 3 m
    pts_cam = src + np.array([0.0, 0.0, 3.0])
    uv, depths = project_pinhole(pts_cam, K)
    print(f"[J7] Projection pinhole : {uv.shape[0]} points dans l'image, "
          f"depth in [{depths.min():.2f}, {depths.max():.2f}] m")

    # Construit une depth map 480x640 a partir des projections (ce que
    # ferait un capteur RGB-D ideal sans bruit). Les pixels non touches
    # restent a 0 = "pas de retour".
    depth_img = np.zeros((480, 640), dtype=np.float32)
    u_pix = np.round(uv[:, 0]).astype(int)
    v_pix = np.round(uv[:, 1]).astype(int)
    valid = (u_pix >= 0) & (u_pix < 640) & (v_pix >= 0) & (v_pix < 480)
    # Si plusieurs points tombent dans le meme pixel, on garde le plus proche
    # — c'est ce que ferait un capteur. Ici simplifie : juste un overwrite.
    depth_img[v_pix[valid], u_pix[valid]] = depths[valid]
    n_filled = int((depth_img > 0).sum())
    print(f"[J7] Depth map : {n_filled} pixels remplis sur {480*640}")

    # ---- 3. Inverse : depth map -> point cloud retrouve ------------------
    pts_recovered = unproject_depth_image(depth_img, K)
    print(f"[J7] Unprojection : {pts_recovered.shape[0]} points recuperes "
          f"(devrait etre proche de {n_filled})")

    # ---- 4. ICP : aligner src perturbe sur target ------------------------
    # Pour que ICP ait du travail, on bruite legerement la source.
    rng = np.random.default_rng(7)
    src_noisy = src + rng.normal(0, 0.005, size=src.shape)

    if HAS_O3D:
        # Version Open3D — plus rapide, KD-tree interne. [Open3D docs]
        pcd_src = o3d.geometry.PointCloud()
        pcd_src.points = o3d.utility.Vector3dVector(src_noisy)
        pcd_tgt = o3d.geometry.PointCloud()
        pcd_tgt.points = o3d.utility.Vector3dVector(target)

        # Voxelise pour homogeneiser la densite (cf. theorie : voxel down)
        pcd_src = pcd_src.voxel_down_sample(0.02)
        pcd_tgt = pcd_tgt.voxel_down_sample(0.02)

        # Threshold = distance maximale pour qu'un voisin compte. Trop petit
        # et ICP croit qu'il n'y a pas de correspondance. Trop grand et il
        # accepte du bruit. 5x voxel_size est un default raisonnable.
        threshold = 0.10
        result = o3d.pipelines.registration.registration_icp(
            pcd_src, pcd_tgt, threshold,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
        )
        T_estimated = np.asarray(result.transformation)
        print(f"[J7] ICP Open3D fitness={result.fitness:.3f} "
              f"inlier_rmse={result.inlier_rmse:.4f}")
    else:
        # Fallback : numpy pur. Plus lent (O(N*M)), reduit la taille.
        idx_src = rng.choice(len(src_noisy), size=min(400, len(src_noisy)),
                             replace=False)
        idx_tgt = rng.choice(len(target), size=min(400, len(target)),
                             replace=False)
        T_estimated, rmse = icp_numpy(src_noisy[idx_src], target[idx_tgt])
        print(f"[J7] ICP numpy : rmse final = {rmse:.4f}")

    # ---- 5. Compare a la verite-terrain ----------------------------------
    # T_estimated devrait approcher la transformation qui envoie src sur
    # target, c'est-a-dire (R_gt, t_gt) en homogene.
    T_gt = np.eye(4)
    T_gt[:3, :3] = R_gt
    T_gt[:3, 3] = t_gt
    err_R = float(np.linalg.norm(T_estimated[:3, :3] - R_gt, ord="fro"))
    err_t = float(np.linalg.norm(T_estimated[:3, 3] - t_gt))
    print(f"[J7] Erreur rotation (Frobenius) : {err_R:.4f}")
    print(f"[J7] Erreur translation         : {err_t:.4f} m")
    print("[J7] Pipeline complet OK.")


if __name__ == "__main__":
    main()
