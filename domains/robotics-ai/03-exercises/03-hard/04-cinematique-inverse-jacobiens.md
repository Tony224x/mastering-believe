# J4 — Exercice HARD : IK redondant 3-DOF avec objectif secondaire (null-space projection)

## Objectif

Exploiter la redondance d'un bras 3-DOF planaire pour atteindre une cible **tout en optimisant un critere secondaire** (eviter les limites articulaires, OU rester loin d'une singularite). Implementer la projection dans le noyau du Jacobien.

## Contexte mathematique

Pour un robot redondant (n > m), parmi les `q_dot` qui realisent une vitesse `x_dot` donnee, on peut choisir librement la composante dans `null(J)`. Formulation classique :

```
q_dot = J^+ x_dot  +  (I - J^+ J) q_dot_secondary
```

ou :
- `J^+ x_dot` est la solution de norme minimale qui realise la tache primaire,
- `(I - J^+ J)` est le **projecteur orthogonal** sur `null(J)`,
- `q_dot_secondary` est le gradient d'un objectif secondaire (a maximiser/minimiser dans le null-space).

Tache primaire (atteindre la cible) toujours satisfaite ; tache secondaire optimisee "best effort".

## Consigne

1. Reprends la FK et le Jacobien 3-DOF de l'exercice MEDIUM.
2. Implemente `ik_dls_with_secondary(fk_fn, jac_fn, target, q_init, q_dot_sec_fn, lam, ...)` :
   ```
   e               = target - fk(q)
   J               = jac(q)
   J_pinv          = J^T (J J^T + lam^2 I)^{-1}        # pseudo-inverse amortie
   dq_primary      = J_pinv @ e
   dq_secondary    = q_dot_sec_fn(q)                    # vecteur de taille n
   null_projector  = I_n - J_pinv @ J
   dq              = dq_primary + null_projector @ dq_secondary
   q               = q + dq
   ```
3. Implemente deux objectifs secondaires :

   **(a) Eviter les limites articulaires.** Pour `q_min, q_max` donnes, le gradient de la "distance au centre" est :
   ```
   q_center = (q_min + q_max) / 2
   q_dot_sec = -k * (q - q_center)
   ```
   avec `k > 0`. Cela tire `q` vers le centre du range admissible.

   **(b) Maximiser la manipulabilite.** Le gradient de `w(q) = sqrt(det(J J^T))` se calcule numeriquement par differences finies (`grad_w = autograd-style numerique`). On pousse `q_dot_sec = +alpha * grad_w` pour s'eloigner des singularites.

4. Scenario de test : limites `q_min = [-pi, -pi, -pi]`, `q_max = [pi, pi, pi]`. Cible `(1.5, 0.0)`. Lance avec et sans objectif secondaire (a) et compare la position finale `q*` : sans, le solveur converge "n'importe ou" dans le null-space ; avec, `q*` sera plus proche du centre `(0, 0, 0)`.

5. **Verification cle** : verifie que la tache primaire (atteindre la cible) reste satisfaite a `tol = 1e-4` meme avec l'objectif secondaire actif.

6. **Question d'analyse** : que se passe-t-il si tu mets un gain `k` trop grand sur l'objectif secondaire ? Et si tu le mets a 0 ? Justifie le compromis.

## Criteres de reussite

- La tache primaire converge (`||target - fk(q*)|| < 1e-4`) avec et sans objectif secondaire.
- Avec objectif (a), `||q* - q_center||` est strictement plus petit qu'avec un solveur DLS standard sur le meme init.
- Le projecteur `(I - J^+ J)` est bien de rang `n - m = 1` (verifie via `np.linalg.matrix_rank`).
- Le code gere proprement le cas `J^+ J ~ I` (peu/pas de redondance) sans diviser par zero.

## Indices

- Le projecteur `(I - J^+ J)` est idempotent : `P @ P == P`. Bonne sanity check.
- `np.linalg.matrix_rank(P, tol=1e-6)` te donne la dimension du null-space.
- Pour le gradient numerique de `w(q) = sqrt(det(J J^T))`, utilise differences finies centrees, pareil que pour le Jacobien numerique.
- Si tu veux pousser plus loin : combine les deux objectifs avec une pondaration (`alpha_a * grad_a + alpha_b * grad_b`).
- Reference : Lynch & Park 2017 §6.3 ("Inverse Velocity Kinematics") et Khatib L8 ("Redundancy Resolution").
