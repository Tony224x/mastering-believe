# J4 — Exercice EASY : IK analytique 2-DOF + sanity check FK

## Objectif

Implementer l'IK closed-form du bras 2-DOF planaire et verifier que `FK(IK(target)) == target` aux deux configurations (coude-haut, coude-bas).

## Consigne

1. Code une fonction `fk_2dof(q, L1, L2)` qui retourne `(x, y)` a partir des angles articulaires.
2. Code `ik_2dof(target, L1, L2, elbow)` qui retourne `(q1, q2)` :
   - utilise la **loi des cosinus** pour `q2`,
   - utilise **`atan2`** (jamais `atan`) pour `q1`,
   - leve une `ValueError` si la cible est hors workspace.
3. Pour `L1 = L2 = 1`, calcule les solutions IK pour les cibles suivantes :
   - `(1.5, 0.5)` (interieur workspace) : verifie les deux configurations elbow.
   - `(2.0, 0.0)` (frontiere workspace, bras tendu).
   - `(3.0, 0.0)` (hors workspace) : verifie que ton code leve `ValueError`.
4. Pour chaque solution, calcule `FK(q)` et imprime `||FK(q) - target||`. L'erreur doit etre `< 1e-10`.

## Criteres de reussite

- Les deux configurations (`up`, `down`) atteignent la meme position cible.
- L'erreur de boucle FK(IK) est inferieure a `1e-10` pour toutes les cibles atteignables.
- La cible hors workspace leve `ValueError` (pas un NaN silencieux).
- `atan2` est utilise (relire ton code : si tu vois `np.arctan` sans `2`, c'est faux).

## Indices

- La distance carree de l'origine a la cible donne directement `cos(q2)` via la loi des cosinus.
- `np.arccos` retourne dans `[0, pi]` ; le coude-haut/bas vient du signe que tu mets devant.
- Pour le quadrant 3 (cible (-x, -y)), `atan2` donne automatiquement le bon angle dans `(-pi, pi]`.
