# Exercice 02-medium — Composition d'une chaine SE(3) et inverse correct

## Objectif

Composer une chaine cinematique 3D (3 reperes le long d'un bras), inverser une transformation rigide de la bonne maniere, et detecter l'erreur classique sur l'inverse.

## Consigne

Ecris un script `02-transformations-3d.py` (scratch local, pas dans le repo) qui :

1. Construit trois transformations homogenes `T_01`, `T_12`, `T_23` :
   - `T_01` : rotation de `pi/4` autour de `z`, translation `(1, 0, 0)`.
   - `T_12` : rotation de `pi/3` autour de `y`, translation `(0, 1, 0)`.
   - `T_23` : rotation identite, translation `(0, 0, 1)`.
   Tu utilises `scipy.spatial.transform.Rotation` pour generer les rotations (axe-angle ou quaternion, ton choix).
2. Calcule la pose composee `T_03 = T_01 @ T_12 @ T_23`. Imprime `T_03`.
3. Implemente deux fonctions :
   - `invert_T_correct(T)` : retourne `[R^T, -R^T p; 0, 1]`.
   - `invert_T_wrong(T)` : retourne `[R^T, -p; 0, 1]` (l'erreur).
4. Verifie que `T_03 @ invert_T_correct(T_03)` est proche de l'identite (`max abs diff < 1e-10`).
5. Calcule `T_03 @ invert_T_wrong(T_03)` et imprime le **max abs diff vs identite**. Cette valeur doit etre **non-nulle** (idealement de l'ordre du module de la translation, ~quelques unites). Commente dans le code pourquoi.
6. Retrouve `T_12` a partir de `T_01` et `T_03` :
   - Sachant que `T_03 = T_01 @ T_12 @ T_23`, exprime `T_12` en fonction de `T_01`, `T_03`, `T_23`. Verifie numeriquement que ta formule retombe bien sur `T_12`.

## Criteres de reussite

- Les deux inverses (correct et wrong) sont implementes.
- `T_03 @ invert_T_correct(T_03) - I` a une norme < 1e-10.
- `T_03 @ invert_T_wrong(T_03) - I` a une norme **clairement non-nulle** (l'exercice doit montrer concretement pourquoi `-p` ne suffit pas).
- La recuperation de `T_12` par algebre matricielle marche (`max abs diff < 1e-10`).

## Hint

Pour la question 6 : `T_12 = T_01^{-1} @ T_03 @ T_23^{-1}`. Pense a multiplier en partant de la gauche par `T_01^{-1}` puis a droite par `T_23^{-1}`. C'est typiquement comment on isole une transformation au milieu d'une chaine cinematique.

Sources : `[Lynch & Park, 2017, §3.3]`, `[Khatib CS223A, L3]`.
