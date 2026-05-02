# J4 — Exercice MEDIUM : IK numerique avec DLS sur un 3-DOF planaire

## Objectif

Implementer une IK numerique generique avec Damped Least Squares, et l'appliquer a un bras 3-DOF planaire (redondant pour une cible 2D). Comparer avec la pseudo-inverse non amortie.

## Consigne

1. Code la FK et le Jacobien analytique d'un bras 3-DOF planaire (`L1=1, L2=1, L3=0.5`). Position d'effecteur en 2D, donc `J` est `2 x 3`.
2. Code une fonction `ik_dls(fk_fn, jac_fn, target, q_init, lam, tol=1e-4, max_iter=200)` qui itere :
   ```
   e  = target - fk(q)
   J  = jac(q)
   dq = J^T (J J^T + lam^2 I)^{-1} e
   q  = q + dq
   ```
   Arret quand `||e|| < tol` ou apres `max_iter` iterations.
3. Cible : `(1.8, 0.4)`. Lance ton solveur trois fois avec :
   - `lam = 0.0` (pseudo-inverse pure),
   - `lam = 0.05` (DLS classique),
   - `lam = 0.5` (DLS fortement amorti).
   Pour chacun, log le nombre d'iterations et la trajectoire `||e||` au cours du temps.
4. **Question** : que se passe-t-il avec `lam=0` quand l'init est tres proche d'une singularite (ex. `q_init = [0.0, 1e-3, 1e-3]`) ? Compare avec `lam=0.05` au meme init.
5. **Bonus** : ajoute un *step-size cap* (borne sur `||dq||` a 0.3 rad) et observe si ca change la convergence.

## Criteres de reussite

- Les 3 solveurs convergent pour `q_init = [0.1, 0.1, 0.1]` sur la cible `(1.8, 0.4)` (interieur workspace).
- Avec `lam=0.0` et init proche singularite, on observe une divergence/oscillation ou des `dq` enormes.
- Avec `lam=0.05`, le solveur converge meme pres de la singularite, eventuellement plus lentement.
- Pour des cibles differentes, plusieurs `q*` distincts sont possibles : c'est la **redondance** (n=3, m=2, dim noyau = 1).

## Indices

- Pour resoudre `(J J^T + lam^2 I) y = e`, utilise `np.linalg.solve` (plus stable que `np.linalg.inv`).
- La trajectoire `||e||` doit decroitre presque monotonement. Si elle oscille, c'est typiquement le step-size.
- La manipulabilite `sqrt(det(J J^T))` te dit si tu es proche d'une singularite : `< 1e-3` = drapeau rouge.
- Pour visualiser la redondance : impose plusieurs `q_init` differents et regarde les `q*` finaux. Ils devraient tous donner la meme `(x, y)` mais des configurations articulaires differentes.
