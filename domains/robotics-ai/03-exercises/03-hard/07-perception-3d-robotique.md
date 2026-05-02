# Exercice J7 — Hard : ICP from scratch + diagnostic de convergence

## Objectif

Reimplementer ICP point-to-point en numpy pur, mesurer experimentalement
son rayon de convergence, et observer le mode d'echec quand l'initialisation
est mauvaise — c'est la cle pour comprendre pourquoi un pipeline industriel
fait toujours global registration AVANT ICP.

## Consigne

### Partie A — implementation

Code une fonction `icp(source, target, max_iter=30) -> (T, history)` ou :

- `source`, `target` sont des `np.ndarray` de shape `(N, 3)` et `(M, 3)`.
- `T` est la matrice 4x4 SE(3) finale.
- `history` est une liste des erreurs RMS apres chaque iteration.

A chaque iteration :

1. Transforme la source courante par la T iterative.
2. Pour chaque point source, trouve son voisin le plus proche dans target
   (KD-tree de scipy, ou matrice de distances vectorisee si N petit).
3. Resout le probleme de Procrustes (alignement rigide de deux nuages
   apparies) par decomposition SVD : `H = src_centered.T @ nn_centered`,
   `U, _, Vt = svd(H)`, `R = Vt.T @ U.T`. Verifie le determinant pour
   eviter une reflexion (`det(R) = -1` est faux, il faut alors flipper le
   signe de la derniere colonne de V).
4. Calcule `t = mean(nn) - R @ mean(src)`.
5. Compose dans T global et reapplique a la source pour la prochaine iter.

### Partie B — rayon de convergence

Genere un point cloud `S` (cube ou sphere, > 500 points). Pour chaque angle
`θ ∈ {5°, 15°, 30°, 60°, 90°, 120°, 180°}` :

1. Construis `target = (R_z(θ) @ S.T).T`.
2. Lance ICP avec `T_init = identity` et `source = S`.
3. Note la RMS finale, le nombre d'iterations, et l'erreur d'angle reelle
   recuperee (compare avec θ vrai).

Trace la courbe `theta_input vs theta_recovered` et `theta_input vs RMS final`.
A partir de quel angle ICP echoue (converge dans un mauvais bassin) ?

### Partie C — global registration en pre-traitement

Pour θ = 120° (cas ou ICP echoue tout seul), implemente un pre-alignement
naif : essaie 12 initialisations de rotation autour de Z (`0°, 30°, 60°,
..., 330°`), garde celle qui donne la meilleure RMS, et raffine avec ICP.
Verifie que tu retrouves la bonne pose.

## Criteres de reussite

- Partie A : ICP converge en < 30 iterations sur deux nuages avec rotation
  initiale < 30° et bruit < 1% du rayon.
- Partie B : tu identifies un seuil empirique au-dela duquel ICP echoue,
  et tu peux l'expliquer (le nearest-neighbor associe les points aux
  mauvais voisins parce que la rotation deplace plus que la moitie des
  echantillons).
- Partie C : la strategie multi-init recupere la bonne pose pour θ = 120°.

## Bonus

Implemente la variante **point-to-plane** (Chen & Medioni 1991). La
fonction objectif n'est plus `sum ||T(p) - q||^2` mais
`sum (n_q · (T(p) - q))^2` ou `n_q` est la normale au point cible. Compare
empiriquement la vitesse de convergence avec point-to-point sur ton banc
d'essai. Tu devrais voir 2-5x moins d'iterations pour la meme precision
finale.
