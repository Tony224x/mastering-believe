# Exercice J7 — Medium : voxelisation et estimation de normales

## Objectif

Maitriser deux pre-traitements indispensables avant ICP : la voxelisation
(downsampling deterministe) et l'estimation des normales (necessaires pour
ICP point-to-plane et pour de nombreux pipelines suivants).

## Consigne

Recupere ou regenere un point cloud dense d'une sphere :

```python
import numpy as np
rng = np.random.default_rng(0)
N = 20000
phi = rng.uniform(0, 2*np.pi, N)
costheta = rng.uniform(-1, 1, N)
theta = np.arccos(costheta)
r = 0.5  # rayon 50 cm
x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)
sphere = np.stack([x, y, z], axis=1)  # (20000, 3)
```

1. **Voxelisation manuelle** : implemente une fonction
   `voxelize(points, voxel_size) -> np.ndarray` qui decoupe l'espace en
   cubes de cote `voxel_size`, garde un seul point representatif par cube
   (par exemple le centroide des points qui tombent dans ce cube), et
   renvoie le nuage downsample.

   *Astuce* : `np.floor(points / voxel_size)` te donne l'index entier du
   voxel de chaque point. Utilise un dict pour grouper, ou la combinaison
   `np.unique(..., return_inverse=True)` + `np.add.at` pour vectoriser.

2. Verifie que voxelizer la sphere a `voxel_size = 0.05` la reduit a
   ~quelques centaines de points (sortie attendue : entre 200 et 800).

3. **Estimation des normales** : pour chaque point du nuage voxelise,
   trouve ses k=15 voisins les plus proches (utilise `scipy.spatial.cKDTree`
   ou Open3D si dispo), calcule la matrice de covariance 3x3 de ces 15
   points, puis prends le vecteur propre associe a la plus petite valeur
   propre. C'est ta normale.

4. Verifie que sur une sphere centree en l'origine, la normale en chaque
   point pointe dans la meme direction que le rayon (au signe pres). Calcule
   `np.abs(np.einsum("ij,ij->i", normals, points / r))`. La moyenne devrait
   etre proche de 1.0 (alignement parfait).

## Criteres de reussite

- La voxelisation produit entre 200 et 800 points pour `voxel_size = 0.05`.
- L'alignement moyen normale/rayon est > 0.95.
- Tu peux expliquer pourquoi le vecteur propre minimal correspond a la
  normale (intuition : la covariance locale est tres "plate" perpendiculairement
  a la surface, et tres etiree dans le plan tangent).

## Bonus

L'estimation des normales avec k=3 voisins donne du bruit ; avec k=100
voisins elle lisse trop la surface. Trace la qualite (alignement moyen)
pour `k ∈ {3, 5, 10, 20, 50, 100}` et discute du compromis.
