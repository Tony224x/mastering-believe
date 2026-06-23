# Exercice 02-easy — Transformations 3D : bases SO(3) / SE(3)

## Objectif

Manipuler une matrice de rotation 3x3 et une transformation homogene 4x4 a la main, et verifier numeriquement les invariants `R^T R = I` et `det(R) = +1`.

## Consigne

Ecris un script `02-transformations-3d.py` (dans un dossier scratch local, pas dans le repo) qui :

1. Construit la matrice de rotation `R = Rx(30 deg) @ Ry(45 deg) @ Rz(60 deg)` ou `Rx`, `Ry`, `Rz` sont les rotations elementaires autour de chaque axe (les ecrire toi-meme, sans utiliser `scipy`).
2. Verifie que `R` est dans SO(3) :
   - `np.allclose(R.T @ R, np.eye(3))` doit etre `True`
   - `np.linalg.det(R)` doit etre proche de `+1` (a 1e-10 pres)
3. Construit une matrice homogene 4x4 `T = [R, p; 0, 1]` avec `p = (1.0, 2.0, 3.0)`.
4. Applique `T` au point `x = (0.5, 0.0, 0.0)` (en homogene) et imprime le point transforme.
5. Affiche les 3 lignes "diagnostic" : `R^T R - I` (max abs), `det(R) - 1`, point transforme.

## Criteres de reussite

- Le script tourne sans erreur sous `python` (pas `python3`).
- Les rotations elementaires `Rx`, `Ry`, `Rz` sont ecrites explicitement (pas via `scipy.spatial.transform`).
- Les deux assertions SO(3) passent (`R^T R = I` ET `det = +1`).
- Le point transforme est bien `R @ x + p`, et tu peux verifier ce calcul a la main mentalement.

## Hint

Une rotation autour de x s'ecrit :

```
       [ 1    0       0    ]
Rx  =  [ 0  cos(t)  -sin(t)]
       [ 0  sin(t)   cos(t)]
```

et de meme pour `Ry`, `Rz` (decaler les colonnes/lignes en consequence).

Source : `[Lynch & Park, 2017, ch. 3]`.
