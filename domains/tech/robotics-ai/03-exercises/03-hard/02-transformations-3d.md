# Exercice 02-hard — Twist, exponentielle se(3), et formule PoE pour 2-DOF

## Objectif

Implementer la formulation Product of Exponentials (PoE) pour un manipulateur 2-DOF planaire et verifier qu'elle redonne le meme resultat que la composition naive de matrices homogenes (la "verite terrain" du cours).

## Consigne

Ecris un script `02-transformations-3d.py` (scratch local) qui :

1. **Twists et exponentielle**. Implemente :
   - `skew(omega)` : matrice antisymetrique 3x3.
   - `twist_to_se3(V)` : `V = (omega, v) ∈ R^6` → matrice 4x4 dans `se(3)`.
   - `exp_se3(V)` : utilise `scipy.linalg.expm` pour calculer `exp([V])`. Tu peux aussi implementer la forme close de Lynch & Park eq. 3.88 si tu veux gagner les bonus points (mais ce n'est pas requis).
2. **Bras planaire 2-DOF — version PoE**. Pour le bras planaire `L1 = L2 = 1` du cours :
   - Configuration home (q = 0) : end-effector a `(L1 + L2, 0, 0)`. Donc `M = [I, (2, 0, 0); 0, 1]`.
   - Le joint 1 tourne autour de l'axe z passant par l'origine. Son screw axis spatial est `S_1 = (omega_1, v_1)` avec `omega_1 = (0, 0, 1)` et `v_1 = -omega_1 × q_1` ou `q_1` est un point sur l'axe (l'origine ici, donc `q_1 = 0` et `v_1 = 0`).
   - Le joint 2 tourne autour de l'axe z passant par `(L1, 0, 0)`. Donc `omega_2 = (0, 0, 1)`, `q_2 = (L1, 0, 0)`, et `v_2 = -omega_2 × q_2 = (0, -L1, 0)`. (Verifier ce signe en croisant avec `[Lynch & Park, 2017, §4.1.2]`.)
   - Implemente `fk_poe(q1, q2)` qui retourne `T = exp([S_1] q1) @ exp([S_2] q2) @ M`.
3. **Bras planaire 2-DOF — version naive (verite terrain)**. Recopie la fonction `fk_planar_2dof` du cours (ou re-implemente la composition `T_w1(q1) @ T_12(q2) @ T_2e`).
4. **Verification**. Pour 5 paires `(q1, q2)` aleatoires (par ex. tirees uniformement dans `[-pi, pi]^2` avec `np.random.seed(42)`), verifie que `fk_poe(q1, q2)` et `fk_naive(q1, q2)` donnent la meme matrice 4x4 a `1e-10` pres.
5. **Bonus** : verifie aussi pour `q = (0, 0)` que les deux donnent bien `[I, (2, 0, 0); 0, 1]`.

## Criteres de reussite

- `exp_se3` valide sur 3 cas connus :
  - twist nul → identite 4x4
  - twist rotation pure (`omega = (0, 0, 1) * pi/2`, `v = 0`) → matrice avec `R = rotz(pi/2)` et `p = 0`
  - twist translation pure (`omega = 0`, `v = (1, 0, 0) * 2`) → matrice avec `R = I` et `p = (2, 0, 0)`
- Pour les 5 configurations aleatoires, `np.max(np.abs(fk_poe - fk_naive)) < 1e-10`.
- Le code commente explicitement le calcul de `v_2 = -omega_2 × q_2` (un point qui pose souvent question : pourquoi le signe moins, pourquoi q_2 et pas l'origine ?).

## Hint mathematique

Le screw axis spatial d'un joint rotoide d'axe `omega` (unitaire) passant par un point `q` (exprime dans le repere espace) est :

```
S = (omega, v)  avec  v = -omega × q
```

Intuition : `v` est la vitesse lineaire du point qui se trouverait a l'origine du repere espace si on tournait a vitesse `omega` autour de l'axe `(omega, q)`. Voir `[Lynch & Park, 2017, §3.3.2]` pour la derivation complete.

## Bonus avance

Compare le temps d'execution de `fk_poe` vs `fk_naive` sur 10000 configurations aleatoires. Lequel est plus rapide ? Pourquoi ? Reflechis a quand chaque approche fait sens (pedagogie / generalisation a n-DOF / performance).

Sources : `[Lynch & Park, 2017, ch. 3-4]`, `[Khatib CS223A, L3]`.
